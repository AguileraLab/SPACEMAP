"""Matplotlib/seaborn figure builders for the tissue annotation pipeline.

Every builder returns a figure so the caller decides whether to render it, save
it, or both.  They live in the science layer rather than in the GUI so that the
Annotation tab and ``notebooks/Step4_annotation.ipynb`` show exactly the same
diagnostics from exactly the same code.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_STYLE = {
    "figure.dpi": 110,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
}


def _apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


# ==========================================================================
# Step 1 - tissue masking
# ==========================================================================

def tissue_mask_figure(result: dict[str, Any], sample_id: str):
    """Stain-strength histogram, virtual H&E preview and the resulting mask."""
    _apply_style()
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    figure.suptitle(f"{sample_id} - H&E optical-density tissue masking", fontsize=14, fontweight="bold")

    stain_score = result.get("he_stain_score")
    valid_pixels = result.get("he_valid_pixels")
    threshold = result["stats"].get("he_od_threshold")
    if stain_score is not None and valid_pixels is not None:
        values = stain_score[valid_pixels][::16]
        axes[0].hist(values, bins=200, color="#888888", alpha=0.75, edgecolor="none")
        if threshold is not None:
            axes[0].axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"OD threshold ({threshold:.3f})")
            axes[0].legend(loc="upper right")
        axes[0].set_xlabel("Mean H&E optical density")
    else:
        axes[0].hist(result["codex_projection"][::8, ::8].ravel(), bins=200, color="#888888", alpha=0.75, edgecolor="none")
        axes[0].set_xlabel("CODEX projection intensity")
    axes[0].set_title("Background vs stain distribution")
    axes[0].set_ylabel("Pixel count")

    he_rgb = result.get("he_rgb")
    he_gray = result.get("he_gray")
    if he_rgb is not None:
        axes[1].imshow(he_rgb, aspect="equal")
    elif he_gray is not None:
        axes[1].imshow(he_gray, cmap="gray", aspect="equal")
    else:
        axes[1].imshow(result["codex_projection"], cmap="gray", aspect="equal")
    axes[1].set_title(f"Virtual H&E ({result['downsample']}x downsampled)")
    axes[1].axis("off")

    axes[2].imshow(result["tissue_mask"], cmap="gray", aspect="equal")
    axes[2].set_title(f"{result['stats']['mode']} mask ({result['stats']['tissue_percent']:.1f}% tissue)")
    axes[2].axis("off")
    return figure


# ==========================================================================
# Step 2 - artifacts
# ==========================================================================

def artifact_figure(
    tissue_mask: np.ndarray,
    artifacts: dict[str, Any],
    codex_robust_max: np.ndarray | None,
    he_gray: np.ndarray | None,
    codex_projection: np.ndarray,
    lint_z_threshold: float,
    sample_id: str,
):
    """Four-panel artifact diagnostic: mask, lint seeds, overlay, clean tissue."""
    import cv2

    _apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    figure.suptitle(f"{sample_id} - blur, bright artifact and external-seed lint diagnostics", fontsize=14, fontweight="bold")
    stats = artifacts["stats"]

    axes[0, 0].imshow(tissue_mask, cmap="gray", aspect="equal")
    axes[0, 0].set_title(f"Reviewed tissue mask ({100.0 * tissue_mask.mean():.1f}% image area)")
    axes[0, 0].axis("off")

    if codex_robust_max is not None:
        vmax = float(np.percentile(codex_robust_max, 99.9))
        axes[0, 1].imshow(codex_robust_max, cmap="magma", vmin=0, vmax=max(vmax, lint_z_threshold))
        contours = cv2.findContours(
            artifacts["lint_mask"].astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        for contour in contours:
            points = contour.reshape(-1, 2)
            axes[0, 1].plot(points[:, 0], points[:, 1], color="#00ffff", linewidth=0.8)
        axes[0, 1].set_title(f"External robust signal and accepted lint (cyan, Z >= {lint_z_threshold:g})")
    else:
        axes[0, 1].imshow(codex_projection, cmap="magma")
        axes[0, 1].set_title("CODEX projection (lint seeds unavailable)")
    axes[0, 1].axis("off")

    if he_gray is not None:
        background_gray = np.clip(255.0 - he_gray, 0, 255).astype(np.uint8)
    else:
        background_gray = np.clip(codex_projection, 0, 255).astype(np.uint8)
    background_rgb = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2RGB)
    overlay = background_rgb.copy()
    overlay[artifacts["blur_mask"]] = [255, 255, 0]
    overlay[artifacts["bright_mask"]] = [255, 0, 255]
    overlay[artifacts["lint_mask"]] = [0, 255, 255]
    blended = cv2.addWeighted(overlay, 0.55, background_rgb, 0.45, 0)
    axes[1, 0].imshow(blended, aspect="equal")
    axes[1, 0].set_title("Detected artifacts")
    axes[1, 0].axis("off")
    axes[1, 0].legend(
        handles=[
            mpatches.Patch(color="#ffff00", label=f"Blurry ({stats['blurry_percent']:.2f}% tissue)"),
            mpatches.Patch(color="#ff00ff", label=f"Bright/flat ({stats['bright_percent']:.2f}% tissue)"),
            mpatches.Patch(color="#00ffff", label=f"External-seed lint ({stats['lint_tissue_percent']:.2f}% overlap)"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    axes[1, 1].imshow(artifacts["valid_tissue_mask"], cmap="gray", aspect="equal")
    axes[1, 1].set_title(f"Clean valid tissue ({stats['valid_percent']:.2f}% retained)")
    axes[1, 1].axis("off")
    return figure


def blur_diagnostic_figure(artifacts: dict[str, Any], sample_id: str):
    """Show why each area was called blurry, and where the threshold falls.

    The histogram is the panel to tune against: it is the distribution of the
    sharpness score over the tissue that was actually judged, with the current
    threshold drawn on it. Everything left of the line is removed. If the line
    sits inside the main body of the distribution rather than under a separate
    left-hand shoulder, the setting is cutting into ordinary tissue.
    """
    _apply_style()
    stats = artifacts["stats"]
    parameters = artifacts.get("parameters", {})
    threshold = float(parameters.get("blur_ratio_threshold", 0.35))
    ratio_map = artifacts.get("blur_ratio_map")
    consensus_map = artifacts.get("consensus_map")
    evaluated = artifacts.get("evaluate_mask")
    if ratio_map is None or evaluated is None:
        return None

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    figure.suptitle(
        f"{sample_id} - out-of-focus scoring ({stats.get('blurry_percent', 0.0):.2f}% of tissue removed)",
        fontsize=14, fontweight="bold",
    )

    values = ratio_map[evaluated]
    values = values[np.isfinite(values)]
    if values.size:
        upper = float(np.percentile(values, 99.5))
        axes[0].hist(np.clip(values, 0, upper), bins=160, color="#888888", alpha=0.8, edgecolor="none")
        axes[0].axvline(threshold, color="#d62728", linestyle="--", linewidth=2,
                        label=f"threshold ({threshold:g})")
        flagged = 100.0 * float((values < threshold).mean())
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].set_title(f"Sharpness score of judged tissue ({flagged:.1f}% below the line)")
    else:
        axes[0].set_title("Sharpness score unavailable")
    axes[0].set_xlabel("Sharpness relative to this image's sharp tissue (1.0 = as sharp as it gets)")
    axes[0].set_ylabel("Pixel count")

    display = np.where(evaluated, np.clip(ratio_map, 0.0, 2.0), np.nan)
    image = axes[1].imshow(display, cmap="RdYlGn", vmin=0.0, vmax=1.5, aspect="equal")
    axes[1].contour(artifacts["blur_mask"].astype(np.float32), levels=[0.5], colors=["#000000"], linewidths=1.0)
    axes[1].set_title("Sharpness map (red = soft, outlined = removed)")
    axes[1].axis("off")
    figure.colorbar(image, ax=axes[1], fraction=0.046, shrink=0.85)

    if consensus_map is not None:
        consensus = np.where(evaluated, consensus_map, np.nan)
        image = axes[2].imshow(consensus, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
        axes[2].set_title("Share of markers out of focus")
        figure.colorbar(image, ax=axes[2], fraction=0.046, shrink=0.85)
    else:
        # Without the per-marker grids the informative thing to show is which
        # tissue was judged at all, since everything else was left untouched.
        skipped = np.zeros((*evaluated.shape, 3), dtype=np.uint8)
        skipped[evaluated] = [70, 70, 70]
        skipped[artifacts["blur_mask"]] = [255, 255, 0]
        axes[2].imshow(skipped, aspect="equal")
        axes[2].set_title(f"Tissue judged for focus ({stats.get('focus_evaluated_percent', 0.0):.1f}%)")
    axes[2].axis("off")
    return figure

# ==========================================================================
# Step 3 - feature engineering and clustering
# ==========================================================================

def valid_tissue_sync_figure(reference_image: np.ndarray, valid_mask: np.ndarray, downsample: int, sample_id: str):
    """Show the reloaded valid-tissue mask on the clustering grid."""
    _apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    figure.suptitle(
        f"{sample_id} - active valid tissue mask (scaled to {downsample}x for clustering)",
        fontsize=13, fontweight="bold",
    )
    axes[0].imshow(reference_image, cmap="gray", aspect="equal")
    axes[0].contour(valid_mask.astype(np.float32), levels=[0.5], colors=["#2ecc71"], linewidths=1.2)
    axes[0].set_title("Valid tissue outline (green)")
    axes[0].axis("off")
    coverage = 100.0 * float(valid_mask.mean())
    axes[1].imshow(valid_mask, cmap="gray", aspect="equal")
    axes[1].set_title(f"Binary mask ({coverage:.1f}% coverage)")
    axes[1].axis("off")
    return figure


def intensity_distribution_figure(feature_stack: dict[str, Any], sample_id: str):
    """Raw vs transformed intensity histograms for the first markers."""
    _apply_style()
    raw = feature_stack["raw_preview"]
    normalized = feature_stack["normalized_preview"]
    names = list(raw.keys())
    if not names:
        return None
    figure, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 6), squeeze=False, constrained_layout=True)
    figure.suptitle("Signal intensity distribution (pre vs post processing)", fontsize=13, fontweight="bold")
    for column, name in enumerate(names):
        cofactor = feature_stack["cofactors"].get(name, "-")
        axes[0][column].hist(raw[name].ravel(), bins=50, color="teal", alpha=0.7)
        axes[0][column].set_title(f"{name} raw")
        axes[0][column].set_yscale("log")
        axes[1][column].hist(normalized[name].ravel(), bins=50, color="coral", alpha=0.7)
        axes[1][column].set_title(f"{name} (cofactor={cofactor})")
        axes[1][column].set_yscale("log")
    return figure


def top_k_elbow_figure(feature_stack: dict[str, Any], config: dict[str, Any], sample_id: str):
    """Z-score spread plus the mean sorted profile used to pick Top-K."""
    _apply_style()
    profile = feature_stack["mean_sorted_profile"]
    marker_names = feature_stack["marker_names"]
    z_valid = feature_stack["pixels_z_valid"]
    effective_k = feature_stack["top_k_effective"]
    auto_k = feature_stack["top_k_auto"]

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    figure.suptitle("Z-score enrichment and Top-K elbow detection", fontsize=13, fontweight="bold")

    first = 0
    second = min(1, len(marker_names) - 1)
    axes[0].hist(z_valid[:, first], bins=50, color="blue", alpha=0.5, label=f"{marker_names[first]} Z-score")
    if second != first:
        axes[0].hist(z_valid[:, second], bins=50, color="red", alpha=0.5, label=f"{marker_names[second]} Z-score")
    axes[0].set_title("Z-score distribution (tissue pixels)")
    axes[0].set_xlabel("Z-score")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)

    selection_mask = feature_stack["selection_mask"]
    if selection_mask.any():
        selected_percent = 100.0 * selection_mask.sum(axis=0) / max(selection_mask.shape[0], 1)
        order = np.argsort(selected_percent)[::-1][:12]
        axes[1].bar([marker_names[index] for index in order], selected_percent[order], color="#4A90E2", alpha=0.85)
        axes[1].set_ylabel("Pixels selected (%)")
        axes[1].set_title("Top-K selection frequency (top 12 markers)")
        axes[1].tick_params(axis="x", rotation=75, labelsize=8)
    else:
        axes[1].text(0.5, 0.5, "Top-K masking disabled", ha="center", va="center")
        axes[1].axis("off")

    ranks = np.arange(1, len(profile) + 1)
    axes[2].plot(ranks, profile, color="#444444", linewidth=2, marker="o", markersize=3)
    axes[2].axvline(effective_k, color="crimson", linestyle="--", linewidth=2, label=f"Selected K={effective_k}")
    if str(config.get("top_k_mode")) != "auto":
        axes[2].axvline(auto_k, color="seagreen", linestyle=":", linewidth=2, label=f"Auto estimate K={auto_k}")
    axes[2].set_title("Mean sorted Z-score profile (elbow)")
    axes[2].set_xlabel("Marker rank (1 = strongest per pixel)")
    axes[2].set_ylabel("Mean Z-score across valid tissue")
    axes[2].legend(fontsize=8)
    return figure


def _clustermap(
    matrix: pd.DataFrame,
    title: str,
    cmap: str,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotation_values: np.ndarray | None = None,
    row_label: str = "Cluster ID",
):
    _apply_style()
    row_cluster = len(matrix) > 1
    grid = sns.clustermap(
        matrix,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        figsize=(16, max(5.0, 0.42 * len(matrix) + 2.5)),
        row_cluster=row_cluster,
        col_cluster=matrix.shape[1] > 1,
        linewidths=0.5,
        linecolor="white",
        dendrogram_ratio=(0.12, 0.10),
        cbar_pos=(0.02, 0.80, 0.03, 0.15),
        yticklabels=[str(value) for value in matrix.index],
        xticklabels=True,
        annot=matrix.to_numpy() if annotation_values is None else annotation_values,
        fmt=".2f",
        annot_kws={"size": 7.0},
    )
    grid.ax_heatmap.set_xlabel("Marker", fontsize=12, fontweight="bold")
    grid.ax_heatmap.set_ylabel(row_label, fontsize=12, fontweight="bold")
    grid.figure.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    return grid


def cluster_signature_figures(profiles: dict[str, Any]) -> dict[str, Any]:
    """The three cluster signature clustermaps plus the balanced row/col order."""
    absolute_grid = _clustermap(
        profiles["absolute_profiles"],
        "Cluster absolute signatures (mean normalized intensity)",
        "YlOrRd",
    )
    z_grid = _clustermap(
        profiles["z_profiles"],
        "Cluster unique/enriched signatures (mean Z-score)",
        "RdBu_r",
        center=0.0,
    )
    limit = float(profiles["color_limit"])
    balanced_grid = _clustermap(
        profiles["balanced_scores"],
        "Cluster size-aware balanced signatures (shrunk mean Z-score)",
        "RdBu_r",
        center=0.0,
        vmin=-limit,
        vmax=limit,
        annotation_values=profiles["shrunk_profiles"].to_numpy(),
    )

    balanced = profiles["balanced_scores"]
    try:
        cluster_order = [balanced.index[i] for i in balanced_grid.dendrogram_row.reordered_ind]
    except Exception:
        cluster_order = list(balanced.index)
    try:
        marker_order = [balanced.columns[i] for i in balanced_grid.dendrogram_col.reordered_ind]
    except Exception:
        marker_order = list(balanced.columns)

    return {
        "absolute": absolute_grid.figure,
        "z_score": z_grid.figure,
        "balanced": balanced_grid.figure,
        "cluster_order": cluster_order,
        "marker_order": marker_order,
    }


def cluster_map_figure(
    cluster_map: np.ndarray,
    cluster_colors: Sequence[str],
    size_summary: pd.DataFrame | None,
    sample_id: str,
    reserved_low_signal_cluster: int | None = None,
):
    """Spatial map of the raw cluster labels over the tissue, with a legend.

    Shown right after clustering so the spatial layout of the clusters can be
    judged before any merging decision, without waiting for the vector export.
    """
    _apply_style()
    active_ids = [int(value) for value in np.unique(cluster_map) if value >= 0]
    if not active_ids:
        return None
    max_id = max(active_ids)
    colormap = mcolors.ListedColormap(["#111111"] + list(cluster_colors))
    bounds = np.arange(-1.5, max_id + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, colormap.N)

    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.imshow(cluster_map, cmap=colormap, norm=norm, interpolation="nearest")
    axis.set_title(f"Pixel clusters on tissue - {sample_id}", fontsize=14, fontweight="bold")
    axis.axis("off")

    handles = []
    for class_id in active_ids:
        color = cluster_colors[min(class_id, len(cluster_colors) - 1)]
        if reserved_low_signal_cluster is not None and class_id == reserved_low_signal_cluster:
            label = "Low signal tissue"
        else:
            label = f"Cluster {class_id}"
        if size_summary is not None and class_id in size_summary.index:
            label += f"  ({size_summary.loc[class_id, 'tissue_percent']:.1f}%)"
        handles.append(mpatches.Patch(color=color, label=label))
    axis.legend(
        handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
        title="Cluster", frameon=True, fontsize=9,
    )
    return figure


def domain_preview_figure(
    cluster_map: np.ndarray,
    cluster_to_domain: dict,
    domain_colors: dict[str, str],
    sample_id: str,
):
    """Spatial preview of the current cluster-to-domain merge.

    Derived from the raw cluster map rather than the exported partition, so it
    updates as soon as assignments change and needs no vectorization.
    """
    _apply_style()
    order: list[str] = []
    for cluster_id in sorted(cluster_to_domain):
        name = str(cluster_to_domain[cluster_id]).strip()
        if name and name not in order:
            order.append(name)
    if not order:
        return None

    domain_index = {name: index for index, name in enumerate(order)}
    label_map = np.full(cluster_map.shape, -1, dtype=np.int16)
    for cluster_id, domain in cluster_to_domain.items():
        domain = str(domain).strip()
        if domain:
            label_map[cluster_map == int(cluster_id)] = domain_index[domain]

    colors = [domain_colors.get(name) or "#888888" for name in order]
    colormap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(order) + 0.5, 1), colormap.N)

    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.imshow(np.ma.masked_equal(label_map, -1), cmap=colormap, norm=norm, interpolation="nearest")
    unassigned = int(np.sum((cluster_map >= 0) & (label_map < 0)))
    title = f"Current merge preview - {sample_id}"
    if unassigned:
        title += f" ({unassigned:,} px still unassigned)"
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.axis("off")
    axis.legend(
        handles=[mpatches.Patch(color=colors[index], label=name) for index, name in enumerate(order)],
        bbox_to_anchor=(1.02, 1), loc="upper left", title="Domain", frameon=True,
    )
    return figure


def cluster_partition_figure(
    cluster_map: np.ndarray,
    export_label_map: np.ndarray,
    cluster_colors: Sequence[str],
    export_mode_label: str,
    sample_id: str,
):
    """Raw cluster labels next to the exact exported gapless partition."""
    _apply_style()
    active_ids = [int(value) for value in np.unique(cluster_map) if value >= 0]
    max_id = max(active_ids) if active_ids else 0
    colormap = mcolors.ListedColormap(["#111111"] + list(cluster_colors))
    bounds = np.arange(-1.5, max_id + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, colormap.N)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    figure.suptitle(f"{sample_id} - pixel classification partition", fontsize=14, fontweight="bold")
    axes[0].imshow(cluster_map, cmap=colormap, norm=norm, interpolation="nearest")
    axes[0].set_title("Original cluster labels")
    axes[0].axis("off")
    axes[1].imshow(export_label_map, cmap=colormap, norm=norm, interpolation="nearest")
    axes[1].set_title(f"Exported partition: {export_mode_label}")
    axes[1].axis("off")
    return figure


def domain_signature_figures(domain_result: dict[str, Any]) -> dict[str, Any]:
    """Signature clustermaps for the merged biological domains."""
    absolute_grid = _clustermap(
        domain_result["domain_profiles"],
        "Merged domain signatures (mean normalized intensity)",
        "YlOrRd",
        row_label="Biological domain",
    )
    z_grid = _clustermap(
        domain_result["domain_z_profiles"],
        "Merged domain unique/enriched signatures (mean Z-score)",
        "RdBu_r",
        center=0.0,
        row_label="Biological domain",
    )
    return {"absolute": absolute_grid.figure, "z_score": z_grid.figure}


def domain_map_figure(domain_result: dict[str, Any], sample_id: str):
    """Final colored spatial map of the merged biological domains."""
    _apply_style()
    order = domain_result["domain_order"]
    colors = domain_result["domain_colors"]
    label_map = domain_result["domain_label_map"]

    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    colormap = mcolors.ListedColormap([colors[name] for name in order])
    bounds = np.arange(-0.5, len(order) + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, colormap.N)
    axis.imshow(np.ma.masked_equal(label_map, -1), cmap=colormap, norm=norm, interpolation="nearest")
    axis.legend(
        handles=[mpatches.Patch(color=colors[name], label=name) for name in order],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Biological domains",
        frameon=True,
    )
    axis.set_title(f"Merged biological domains - {sample_id}", fontsize=14, fontweight="bold")
    axis.axis("off")
    return figure
