from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spacemap.acc_utils import DEFAULT_PHENOTYPE_COLORS
from spacemap_gui.tabs.tab9_visualization.phenotype_colors import (
    add_category_color_strip,
    apply_category_tick_colors,
    canonical_color_key,
    category_labels_for_columns,
    shared_phenotype_colors,
)
from spacemap_gui.tabs.tab9_visualization.proportion import _make_proportion_figure
from spacemap_gui.tabs.tab9_visualization.umap_explorer import _make_umap_figure_categorical


def test_shared_phenotype_colors_use_house_defaults_and_preserve_overrides() -> None:
    state = {canonical_color_key("CD4"): "#123456"}

    colors = shared_phenotype_colors(["CD8", "CD4", "Tumor"], state)

    assert colors["CD4"] == "#123456"
    assert colors["CD8"] == DEFAULT_PHENOTYPE_COLORS["CD8"]
    assert state[canonical_color_key("Tumor")] == colors["Tumor"]


def test_shared_phenotype_colors_repairs_legacy_all_black_palette_once() -> None:
    state = {
        canonical_color_key("CD4"): "#000000",
        canonical_color_key("CD8"): "#000000",
    }

    colors = shared_phenotype_colors(["CD4", "CD8"], state)

    assert colors == {
        "CD4": DEFAULT_PHENOTYPE_COLORS["CD4"],
        "CD8": DEFAULT_PHENOTYPE_COLORS["CD8"],
    }


def test_category_labels_for_columns_combines_selected_phenotyping_methods() -> None:
    frame = pd.DataFrame(
        {
            "phenotype_a": ["CD4", "CD8", None],
            "phenotype_b": ["Tumor", "CD4", "  "],
        }
    )

    assert category_labels_for_columns(frame, ["phenotype_a", "phenotype_b"]) == [
        "CD4",
        "CD8",
        "Tumor",
    ]


def test_heatmap_category_tick_labels_receive_shared_colors() -> None:
    figure, axis = plt.subplots()
    axis.set_xticks([0, 1], labels=["CD4", "Tumor"])

    apply_category_tick_colors(axis, {"CD4": "#123456", "Tumor": "#abcdef"}, which="x")
    add_category_color_strip(
        axis,
        ["CD4", "Tumor"],
        {"CD4": "#123456", "Tumor": "#abcdef"},
        which="x",
    )

    assert [to_hex(label.get_color()) for label in axis.get_xticklabels()] == ["#123456", "#abcdef"]
    assert [to_hex(patch.get_facecolor()) for patch in axis.patches] == ["#123456", "#abcdef"]
    plt.close(figure)


def test_umap_categorical_figure_uses_the_shared_mapping() -> None:
    figure = _make_umap_figure_categorical(
        umap_x=np.asarray([0.0, 1.0]),
        umap_y=np.asarray([0.0, 1.0]),
        labels=np.asarray(["CD4", "Tumor"]),
        palette_name="tab10",
        custom_color_map={"CD4": "#123456", "Tumor": "#abcdef"},
        point_size=5.0,
        alpha=1.0,
        fig_width=4.0,
        fig_height=4.0,
        dpi=100,
        title="test",
        show_legend=True,
        legend_fontsize=8,
        legend_ncol=1,
        legend_markerscale=1.0,
        label_fontsize=10,
        title_fontsize=12,
        axis_off=False,
        rasterized=False,
        background_color="white",
        grid_alpha=0.0,
        order=None,
    )

    plotted_colors = [to_hex(collection.get_facecolors()[0]) for collection in figure.axes[0].collections]
    assert plotted_colors == ["#123456", "#abcdef"]
    plt.close(figure)


def test_proportion_bars_are_colored_by_phenotype_not_method() -> None:
    summary = pd.DataFrame(
        {
            "method": ["phenotype_a", "phenotype_a", "phenotype_b", "phenotype_b"],
            "phenotype": ["CD4", "Tumor", "CD4", "Tumor"],
            "proportion": [0.4, 0.6, 0.3, 0.7],
        }
    )
    figure = _make_proportion_figure(
        summary,
        methods=["phenotype_a", "phenotype_b"],
        phenotype_order=["CD4", "Tumor"],
        phenotype_colors={"CD4": "#123456", "Tumor": "#abcdef"},
        figsize=(8.0, 4.0),
    )

    patch_colors = [to_hex(patch.get_facecolor()) for patch in figure.axes[0].patches]
    assert patch_colors == ["#123456", "#abcdef", "#123456", "#abcdef"]
    assert figure.axes[0].patches[0].get_hatch() != figure.axes[0].patches[2].get_hatch()
    plt.close(figure)
