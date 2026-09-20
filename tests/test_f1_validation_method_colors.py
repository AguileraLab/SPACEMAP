from __future__ import annotations

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import pandas as pd

from spacemap_gui.tabs.tab9_visualization.f1_paper_figures import METHOD_COLORS
from spacemap_gui.tabs.tab9_visualization.f1_validation_utils import (
    GROUP_POOLED,
    OPTION2_METHOD_COLORS,
    ValidationStyle,
    _method_palette,
    plot_f1_metric_comparison,
)


def test_option1_semantic_defaults_match_option2_exactly() -> None:
    assert OPTION2_METHOD_COLORS == METHOD_COLORS
    methods = [
        "phenotype_leiden",
        "phenotype_som",
        "phenotype_vortex",
        "phenotype_threshold",
        "prediction_mr_ml_strict",
        "prediction_consensus_ml_strict",
    ]

    palette = _method_palette(methods, ValidationStyle())

    assert palette == {
        "phenotype_leiden": METHOD_COLORS["Leiden"],
        "phenotype_som": METHOD_COLORS["SOM"],
        "phenotype_vortex": METHOD_COLORS["Vortex"],
        "phenotype_threshold": METHOD_COLORS["RESOLVE"],
        "prediction_mr_ml_strict": METHOD_COLORS["ML-MR"],
        "prediction_consensus_ml_strict": METHOD_COLORS["Consensus-ML"],
    }


def test_option1_method_colors_are_stable_when_order_changes() -> None:
    style = ValidationStyle()
    forward = _method_palette(["Leiden", "SOM", "Vortex"], style)
    reverse = _method_palette(["Vortex", "SOM", "Leiden"], style)

    assert forward["Leiden"] == reverse["Leiden"] == METHOD_COLORS["Leiden"]
    assert forward["SOM"] == reverse["SOM"] == METHOD_COLORS["SOM"]
    assert forward["Vortex"] == reverse["Vortex"] == METHOD_COLORS["Vortex"]


def test_option1_custom_method_colors_reach_plotted_artists() -> None:
    summary = pd.DataFrame(
        {
            "reference": ["Manual", "Manual"],
            "route": ["End-to-end", "End-to-end"],
            "group": [GROUP_POOLED, GROUP_POOLED],
            "method": ["Leiden", "Consensus-ML"],
            "micro_f1": [0.8, 0.9],
            "macro_f1": [0.75, 0.85],
            "weighted_f1": [0.78, 0.88],
        }
    )
    style = ValidationStyle(
        method_colors={"Leiden": "#123456", "Consensus-ML": "#abcdef"},
        figure_width=6.0,
        figure_height=3.0,
    )

    figure = plot_f1_metric_comparison(
        {"summary": summary},
        reference="Manual",
        route="End-to-end",
        style=style,
        method_order=["Leiden", "Consensus-ML"],
    )
    try:
        plotted = [to_hex(patch.get_facecolor()) for patch in figure.axes[0].patches]
    finally:
        plt.close(figure)

    assert plotted == ["#123456", "#abcdef"]
