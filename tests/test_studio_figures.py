"""Publication figures: a clustered heatmap or classic's dot plot of any grouping, as PNG, SVG or PDF."""

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from spacemap_gui.studio import figures
from test_studio_commands import setup  # noqa: F401


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    rng = np.random.default_rng(3)
    parts = []
    for group, marker in (("Cluster 0", "CD3"), ("Cluster 1", "PanCK"), ("Cluster 2", "CD68")):
        frame = pd.DataFrame({m: rng.gamma(2.0, 20.0, 200) for m in ("CD3", "PanCK", "CD68")})
        frame[marker] += rng.gamma(20.0, 60.0, 200)
        frame["CD3_corr"] = np.arcsinh(frame["CD3"] / 5)
        frame["phenotype_x_clusters"] = group
        parts.append(frame)
    path = tmp_path / "combined.csv"
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return path


def test_a_heatmap_and_a_dot_plot_of_the_groups(table):
    png, kind = figures.table_figure(
        table, "phenotype_x_clusters", ["CD3", "PanCK", "CD68"], "heatmap", {"title": "Leiden"}
    )
    assert kind == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n"
    names = {"Cluster 0": "T cell", "Cluster 1": "Tumour", "Cluster 2": "Macrophage"}
    svg, kind = figures.table_figure(
        table, "phenotype_x_clusters", ["CD3_corr", "PanCK"], "dotplot", {"format": "svg"}, names=names
    )
    assert kind == "image/svg+xml" and b"<svg" in svg[:400] and b"Tumour" in svg
    pdf, kind = figures.matrix_figure(
        ["CN 0", "CN 1", "CN 2"], ["T cell", "Tumour"], [[1, -1], [0, 2], [-1, 0]], "Neighbourhoods", {"format": "pdf"}
    )
    assert kind == "application/pdf" and pdf[:4] == b"%PDF"


def test_wrong_requests_are_refused(table):
    for call in (
        lambda: figures.table_figure(table, "nope", ["CD3"], "heatmap"),
        lambda: figures.table_figure(table, "phenotype_x_clusters", [], "heatmap"),
        lambda: figures.table_figure(table, "phenotype_x_clusters", ["CD3"], "pie"),
        lambda: figures.table_figure(table, "phenotype_x_clusters", ["CD3"], "heatmap", {"format": "exe"}),
        lambda: figures.table_figure(table, "phenotype_x_clusters", ["CD3"], "heatmap", {"dpi": 5000}),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("aggregation", ["median", "mean"])
def test_full_table_heatmap_uses_every_cell_and_does_not_rescale_corrected_values(table, monkeypatch, aggregation):
    from spacemap.cluster_profiles import standardize_groups
    frame = pd.read_csv(table)
    frame["roi"] = ["first_image"] * 210 + ["another_image"] * 390
    frame.to_csv(table, index=False)
    monkeypatch.setattr(figures, "PER_GROUP", 3)
    captured = {}
    monkeypatch.setattr(figures, "_clustermap", lambda matrix, title, options: captured.update(matrix=matrix, title=title))
    monkeypatch.setattr(figures, "_save", lambda *args: b"plot")
    figures.table_figure(table, "phenotype_x_clusters", ["CD3_corr"], "heatmap", {"aggregation":aggregation})
    expected = standardize_groups(frame.groupby("phenotype_x_clusters")[["CD3_corr"]].agg(aggregation))
    expected.columns = ["CD3"]
    pd.testing.assert_frame_equal(captured["matrix"], expected, check_names=False)
    assert "all 600 labelled cells" in captured["title"]


@pytest.mark.parametrize("x,y", [(False,False), (True,False), (False,True), (True,True)])
@pytest.mark.parametrize("transpose", [False, True])
def test_axes_are_independent_and_colour_bar_cannot_overlap_labels(x, y, transpose):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    matrix = pd.DataFrame([[0, 9, 2], [8, 0, 1], [1, 8, 2], [9, 1, 0]],
        index=["A very long phenotype label for cluster " + str(i) for i in range(4)],
        columns=["A long marker name _corrected_" + str(i) for i in range(3)])
    fig = figures._clustermap(matrix, "A publication heatmap with long labels", {
        "cluster_x":x, "cluster_y":y, "groups_on":"columns" if transpose else "rows"})
    shown = matrix.T if transpose else matrix
    ri = leaves_list(linkage(shown.to_numpy(), method="average")) if y else list(range(len(shown)))
    ci = leaves_list(linkage(shown.to_numpy().T, method="average")) if x else list(range(len(shown.columns)))
    pd.testing.assert_frame_equal(fig.spacemap_matrix, shown.iloc[ri,ci])
    axes = {ax.get_label():ax for ax in fig.axes}
    assert ("row dendrogram" in axes) == y and ("column dendrogram" in axes) == x
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    heat, bar = axes["heatmap"].get_window_extent(renderer), axes["colour scale"].get_window_extent(renderer)
    assert bar.x0 > heat.x1
    for label in axes["heatmap"].get_yticklabels():
        assert label.get_window_extent(renderer).x1 < heat.x0
    for label in axes["heatmap"].get_xticklabels():
        assert label.get_window_extent(renderer).y1 < heat.y0
    assert fig._suptitle.get_window_extent(renderer).y0 > heat.y1
    plt.close(fig)


def test_run_publication_uses_classic_named_profiles_for_all_cells(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import cluster_profiles
    from spacemap_gui.viewer.api import app
    from test_studio_chat import _job
    from spacemap.cluster_profiles import named_profiles
    channel, table, root, _ = setup
    frame = pd.DataFrame({"cluster":[0,0,0,1,2,2,2,2], "M0_norm":[1,2,100,4,8,9,10,11], "M1_norm":[7,8,9,10,1,2,3,4]})
    clusters, profile = cluster_profiles.build(frame, ["M0_norm","M1_norm"], "cluster", config={"norm_z1":True})
    job_id = _job(root, "leiden", {"table":str(table), "column":"phenotype_x_clusters", "cells":8,
        "clusters":clusters,"profile":profile})
    captured = {}
    monkeypatch.setattr(figures, "_clustermap", lambda matrix, title, options: captured.update(matrix=matrix,title=title))
    monkeypatch.setattr(figures, "_save", lambda *args: b"plot")
    names = {"Cluster 0":"T cells", "Cluster 1":"T cells", "Cluster 2":"Stroma"}
    response = TestClient(app).post(f"/studio/{channel}/figure", json={
        "source":"run", "job_id":job_id, "kind":"heatmap", "names":names, "options":{"aggregation":"mean"},
        "image":"first_image_only", "selected_cells":[1]})
    assert response.status_code == 200, response.text
    mean = profile["summaries"]["mean"]
    expected = named_profiles(pd.DataFrame(mean["values"], index=mean["rows"], columns=mean["columns"]), names)
    pd.testing.assert_frame_equal(captured["matrix"], expected)
    assert "Full saved run · 8 cells" in captured["title"] and "Mean ·" in captured["title"] and "Z1" in captured["title"]
    with pytest.raises(ValueError, match="another combined table"):
        figures.run_figure(root, "other.csv", job_id, names)


def test_table_api_reads_combined_table_regardless_of_image_selection(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.viewer.api import app
    channel, table, _, _ = setup
    captured = {}
    monkeypatch.setattr(figures, "_clustermap", lambda matrix, title, options: captured.update(matrix=matrix,title=title))
    monkeypatch.setattr(figures, "_save", lambda *args: b"plot")
    response=TestClient(app).post(f"/studio/{channel}/figure",json={"source":"table", "column":"phenotype_ml_consensus",
        "markers":["M0"],"kind":"heatmap", "selected_cells":[1], "roi":"P11_roi_2"})
    assert response.status_code == 200, response.text
    frame=pd.read_csv(table)
    assert f"all {frame.phenotype_ml_consensus.notna().sum():,} labelled cells" in captured["title"]
