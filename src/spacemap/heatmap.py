"""Publication heatmaps shared by classic and Studio, with space reserved for every axis."""

import textwrap

import numpy as np


def publication_heatmap(matrix, title, options=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from scipy.cluster.hierarchy import dendrogram, linkage
    import seaborn as sns

    options = options or {}
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.loc[matrix.notna().any(axis=1), matrix.notna().any(axis=0)].fillna(0)
    if matrix.empty:
        raise ValueError("There are no values to draw.")
    if options.get("groups_on") == "columns":
        matrix = matrix.T
    # X and Y always refer to the displayed axes, including when transposed.
    cluster_y = bool(options.get("cluster_y", options.get("cluster", True))) and len(matrix) > 1
    cluster_x = bool(options.get("cluster_x", options.get("cluster", True))) and len(matrix.columns) > 1
    row_tree = linkage(matrix.to_numpy(), method="average") if cluster_y else None
    col_tree = linkage(matrix.to_numpy().T, method="average") if cluster_x else None
    row_order = dendrogram(row_tree, no_plot=True)["leaves"] if cluster_y else list(range(len(matrix)))
    col_order = dendrogram(col_tree, no_plot=True)["leaves"] if cluster_x else list(range(len(matrix.columns)))
    shown = matrix.iloc[row_order, col_order]

    # Measure at the actual font size rather than estimating margins from character counts.
    fig = plt.figure(dpi=100)
    renderer = FigureCanvasAgg(fig).get_renderer()
    font = FontProperties(size=9)
    width = lambda labels: max((renderer.get_text_width_height_descent(str(s), font, False)[0] / fig.dpi for s in labels), default=0)
    label_width, marker_width = width(shown.index), width(shown.columns)
    heat_w, heat_h = max(2.0, .32 * len(shown.columns)), max(2.0, .28 * len(shown))
    tree_w, tree_h = (.65 if cluster_y else 0), (.65 if cluster_x else 0)
    left = .25 + tree_w + label_width + .22
    bottom = .25 + marker_width * np.sin(np.pi / 3) + .22
    fig_w = left + heat_w + 1.35
    wrapped = "\n".join(textwrap.fill(line, max(25, int(fig_w * 9))) for line in str(title).splitlines())
    title_h = .3 + .2 * max(1, len(wrapped.splitlines()))
    fig_h = bottom + heat_h + tree_h + title_h
    fig.set_size_inches(fig_w, fig_h)
    axes = lambda x, y, w, h, label: fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h], label=label)
    ax = axes(left, bottom, heat_w, heat_h, "heatmap")
    cbar_h = min(heat_h, 3.0)
    cax = axes(left + heat_w + .35, bottom + (heat_h - cbar_h) / 2, .17, cbar_h, "colour scale")
    cmap = options.get("cmap", "RdBu_r")
    limit = float(np.max(np.abs(shown.to_numpy()))) or 1.0
    low, high = options.get("vmin"), options.get("vmax")
    low, high = -limit if low is None else float(low), limit if high is None else float(high)
    if not np.isfinite([low, high]).all() or low >= high:
        plt.close(fig)
        raise ValueError("The colour scale needs a finite minimum below its maximum.")
    sns.heatmap(shown, ax=ax, cbar_ax=cax, cmap=cmap, vmin=low, vmax=high,
                center=0 if cmap not in ("viridis", "magma") else None,
                xticklabels=True, yticklabels=True,
                cbar_kws={"label": options.get("value_label", "Marker z-score")})
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=9, length=0, pad=6)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    cax.tick_params(labelsize=9)
    cax.yaxis.set_label_position("right"); cax.yaxis.set_ticks_position("right")
    if cluster_y:
        tree = axes(.15, bottom, tree_w, heat_h, "row dendrogram")
        dendrogram(row_tree, orientation="left", ax=tree, no_labels=True, color_threshold=0, above_threshold_color="#555555")
        tree.invert_yaxis(); tree.set_axis_off()
    if cluster_x:
        tree = axes(left, bottom + heat_h + .05, heat_w, tree_h - .05, "column dendrogram")
        dendrogram(col_tree, ax=tree, no_labels=True, color_threshold=0, above_threshold_color="#555555")
        tree.set_axis_off()
    fig.suptitle(wrapped, x=(left + heat_w / 2) / fig_w, y=1 - .1 / fig_h, fontsize=12, va="top")
    # Useful to exporters and regression checks; never used to select cells.
    fig.spacemap_matrix = shown
    return fig
