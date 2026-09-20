"""Subtab 3 - pixel classification of the reviewed valid tissue.

Normalizes the marker stack, clusters valid-tissue pixels, characterises the
clusters with three complementary signature heatmaps, and lets the user merge
clusters into biological domains by clicking the size-aware balanced heatmap.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import streamlit as st

from spacemap_gui.process_log import record_process_event

from . import figures
from .common import (
    TAB_CONTEXT,
    describe_output,
    geojson_path,
    log_annotation,
    open_reader,
    output_dir_for,
    overlay_polygons_on_thumbnail,
    qupath_hint,
    prune_widget_selection,
    render_class_legend,
    render_image_preview,
    render_parameter_form,
    resolve_image_path,
    runtime_store,
    save_config,
    save_project_settings,
    save_table,
    show_cached_figure,
    sticky_file_selectbox,
)

CONFIG_KEY = "annot_ta_cluster_config"
MARKERS_KEY = "annot_ta_cluster_markers"
DOMAIN_MAP_KEY = "annot_ta_domain_map"
DOMAIN_NAMES_KEY = "annot_ta_domain_names"
DOMAIN_COLORS_KEY = "annot_ta_domain_colors"
EXCLUDED_KEY = "annot_ta_excluded_clusters"

UNASSIGNED = "Unassigned"

# Common tissue domain names offered as a starting catalogue. They are kept
# short and generic on purpose: these become QuPath classification names and
# feed the cell-annotation column names downstream. Anything missing can be
# typed in and is saved with the project.
COMMON_DOMAIN_LABELS: dict[str, list[str]] = {
    "Structural": [
        "Epithelia", "Stroma", "Endothelia", "Muscle", "Smooth_Muscle",
        "Adipose", "Nerve", "Capsule", "Basement_Membrane",
    ],
    "Glandular and luminal": [
        "Acinar", "Ductal", "Alveolar", "Lumen", "Crypt", "Villi", "Follicle", "Secretory",
    ],
    "Tumor": [
        "Tumor", "Tumor_Core", "Tumor_Margin", "Invasive_Front", "Dysplasia", "Metastasis",
    ],
    "Immune": [
        "Immune", "TLS", "Lymphoid_Aggregate", "Germinal_Center",
        "Myeloid", "Lymphocyte_Rich", "Inflammation",
    ],
    "Vascular": [
        "Vessel", "Blood_Vessel", "Lymphatic", "Perivascular",
    ],
    "Other": [
        "Necrosis", "Fibrosis", "Hemorrhage", "Mucin", "Keratin",
        "Low_Signal_Tissue", "Artifact", "Background",
    ],
}

ALL_COMMON_DOMAIN_LABELS = [label for group in COMMON_DOMAIN_LABELS.values() for label in group]


def _cluster_spatial_summary(cluster_map: np.ndarray, cluster_ids: Sequence[int]) -> pd.DataFrame:
    """Make a small numeric description of the spatial cluster map for AI review."""
    array = np.asarray(cluster_map)
    if array.ndim != 2 or array.size == 0:
        return pd.DataFrame()
    height, width = array.shape
    known = {int(value) for value in cluster_ids}
    adjacent: dict[int, set[int]] = {value: set() for value in known}
    for left, right in (
        (array[:, :-1], array[:, 1:]),
        (array[:-1, :], array[1:, :]),
    ):
        changed = left != right
        if not np.any(changed):
            continue
        pairs = np.column_stack((left[changed], right[changed]))
        for first, second in np.unique(pairs, axis=0):
            try:
                first_int, second_int = int(first), int(second)
            except (TypeError, ValueError, OverflowError):
                continue
            if first_int in known and second_int in known and first_int != second_int:
                adjacent[first_int].add(second_int)
                adjacent[second_int].add(first_int)

    rows = []
    for cluster in sorted(known):
        ys, xs = np.where(array == cluster)
        if not len(xs):
            continue
        x_percent = 100.0 * float(xs.mean()) / max(1, width - 1)
        y_percent = 100.0 * float(ys.mean()) / max(1, height - 1)
        horizontal = "left" if x_percent < 33.3 else "right" if x_percent > 66.7 else "center"
        vertical = "top" if y_percent < 33.3 else "bottom" if y_percent > 66.7 else "middle"
        touches_edge = bool(
            np.any(xs == 0)
            or np.any(xs == width - 1)
            or np.any(ys == 0)
            or np.any(ys == height - 1)
        )
        rows.append(
            {
                "Cluster": cluster,
                "Map location": f"{vertical}-{horizontal}",
                "Centroid X (%)": round(x_percent, 1),
                "Centroid Y (%)": round(y_percent, 1),
                "Horizontal span (%)": round(
                    100.0 * float(xs.max() - xs.min() + 1) / max(1, width), 1
                ),
                "Vertical span (%)": round(
                    100.0 * float(ys.max() - ys.min() + 1) / max(1, height), 1
                ),
                "Touches map edge": touches_edge,
                "Adjacent clusters": ", ".join(str(value) for value in sorted(adjacent[cluster])),
            }
        )
    return pd.DataFrame(rows)

PARAMETER_SPEC = [
    {"type": "header", "label": "Resolution", "help": "Grid the clustering and the exported polygons live on."},
    {
        "key": "clustering_downsample", "label": "Clustering downsample (x)", "type": "int", "min": 1, "max": 32, "step": 1,
        "help": (
            "Pyramid level used for the marker feature stack and for the exported polygon grid. "
            "4 gives roughly 2 um per raster pixel at 0.5083 um/px. Raise it for very large ROIs or "
            "limited memory; the exported boundaries become correspondingly coarser."
        ),
    },
    {
        "key": "microns_per_pixel", "label": "Pixel size (um/px at full res)", "type": "number",
        "min": 0.0001, "max": 10.0, "step": 0.0001, "format": "%.4f",
        "help": "Full-resolution pixel size, used to convert the micron-based export tolerances.",
    },
    {"key": "random_seed", "label": "Random seed", "type": "int", "min": 0, "max": 1_000_000, "step": 1,
     "help": "Seed for MiniBatchKMeans and for the deterministic domain colors. Keep it fixed for reproducible runs."},

    {"type": "header", "label": "Signal transformation",
     "help": "Per-marker arcsinh, robust scaling and light spatial smoothing before Z-scoring."},
    {
        "key": "arcsinh_mode", "label": "Arcsinh mode", "type": "select", "options": ["dynamic", "fixed"],
        "help": (
            "dynamic estimates each marker's background noise inside tissue and sets its cofactor to "
            "max(minimum, k x sigma_background), so noisy channels are compressed harder. "
            "fixed applies one cofactor to every marker."
        ),
    },
    {"key": "arcsinh_cofactor", "label": "Fixed cofactor", "type": "number", "min": 0.1, "max": 1000.0, "step": 0.5,
     "help": "Cofactor used in fixed mode, and as the fallback when a marker has too few tissue pixels to estimate noise."},
    {
        "key": "arcsinh_dynamic_k", "label": "Dynamic k (noise multiplier)", "type": "number",
        "min": 0.1, "max": 20.0, "step": 0.1,
        "help": (
            "Signal-to-noise floor. With background sigma 2.0 and k=3.0 the cofactor becomes 6.0, so intensities "
            "below 6 are compressed as noise while higher values scale close to linearly."
        ),
    },
    {"key": "arcsinh_dynamic_min", "label": "Dynamic cofactor floor", "type": "number", "min": 0.1, "max": 1000.0, "step": 0.5,
     "help": "Lower bound for the dynamic cofactor, which prevents very clean channels from getting an unstably small value."},
    {"key": "percentile_threshold", "label": "Upper clip percentile", "type": "number", "min": 50.0, "max": 100.0, "step": 0.1,
     "help": "Per-channel percentile used as the upper bound when rescaling to 0-1. Clips hot pixels."},
    {"key": "sigma_spatial_smoothing", "label": "Spatial smoothing sigma", "type": "number", "min": 0.0, "max": 10.0, "step": 0.1,
     "help": "Gaussian smoothing per channel on the clustering grid, which suppresses pixel-level shot noise. 0 disables it."},

    {"type": "header", "label": "Feature selection",
     "help": "Whether each pixel keeps only its most enriched markers before clustering."},
    {
        "key": "use_top_k_masking", "label": "Use Top-K marker masking", "type": "checkbox",
        "help": (
            "Keep only the K strongest marker Z-scores per pixel and zero the rest. This sharpens cluster "
            "identity when a panel has many weakly co-expressed markers. Uncheck to cluster on continuous Z-scores."
        ),
    },
    {
        "key": "top_k_mode", "label": "Top-K mode", "type": "select", "options": ["auto", "manual"],
        "help": (
            "auto finds the elbow of the mean sorted Z-score profile, which adapts K to how many markers are "
            "typically co-enriched in this sample and panel. manual always uses the fixed K below."
        ),
    },
    {"key": "top_k_markers", "label": "Manual K", "type": "int", "min": 1, "max": 60, "step": 1,
     "help": "K used in manual mode, and the fallback if automatic detection fails."},
    {"key": "top_k_auto_min", "label": "Auto K minimum", "type": "int", "min": 1, "max": 60, "step": 1,
     "help": "Smallest K the elbow search may pick."},
    {"key": "top_k_auto_max", "label": "Auto K maximum", "type": "int", "min": 1, "max": 60, "step": 1,
     "help": "Largest K the elbow search may pick."},
    {
        "key": "low_signal_handling", "label": "Low-signal pixels", "type": "select", "options": ["include", "reserve"],
        "help": (
            "include clusters every valid-tissue pixel. reserve sets aside the weakest pixels as a separate "
            "Low_Signal_Tissue class so they cannot distort the marker-driven clusters."
        ),
    },
    {"key": "low_signal_quantile", "label": "Low-signal quantile", "type": "number", "min": 0.0, "max": 0.5, "step": 0.01,
     "help": "Fraction of the weakest tissue pixels reserved when low-signal handling is set to reserve."},

    {"type": "header", "label": "Clustering", "help": "MiniBatchKMeans settings."},
    {"key": "n_clusters", "label": "Number of clusters (k)", "type": "int", "min": 2, "max": 100, "step": 1,
     "help": "How many pixel classes to discover. Over-clustering is normally safer: neighbouring clusters can be merged into one domain afterwards."},
    {"key": "kmeans_batch_size", "label": "Mini-batch size", "type": "int", "min": 64, "max": 100_000, "step": 64,
     "help": "Pixels per MiniBatchKMeans update. Larger batches are more stable but slower per iteration."},
    {"key": "kmeans_n_init", "label": "Initializations", "type": "int", "min": 1, "max": 50, "step": 1,
     "help": "Number of random restarts. The best inertia is kept."},
    {"key": "kmeans_max_iter", "label": "Maximum iterations", "type": "int", "min": 10, "max": 5000, "step": 10,
     "help": "Iteration cap per initialization."},
    {
        "key": "kmeans_fit_sample", "label": "Fit subsample (pixels, 0 = all)", "type": "int",
        "min": 0, "max": 50_000_000, "step": 100_000,
        "help": (
            "Fit the centroids on a random subsample of tissue pixels and then assign every pixel. "
            "On very large ROIs this cuts run time substantially with almost identical centroids. "
            "0 fits on all valid-tissue pixels."
        ),
    },

    {"type": "header", "label": "Size-aware balanced heatmap",
     "help": "Controls the interactive merge heatmap, so neither tiny nor dominant clusters distort the colors."},
    {
        "key": "size_shrinkage_prior_fraction", "label": "Shrinkage prior (fraction of median size)", "type": "number",
        "min": 0.0, "max": 5.0, "step": 0.05,
        "help": (
            "Reliability weight prior. A cluster with pixels equal to this prior is shrunk halfway toward the "
            "tissue-wide marker mean, which stops scarce clusters from showing extreme, unstable signatures."
        ),
    },
    {"key": "balanced_marker_quantile", "label": "Marker color-scale quantile", "type": "number", "min": 0.5, "max": 1.0, "step": 0.01,
     "help": "Per-marker robust scale used to normalize color contrast across markers."},
    {"key": "balanced_min_z_scale", "label": "Minimum marker Z scale", "type": "number", "min": 0.01, "max": 5.0, "step": 0.05,
     "help": "Floor on that scale, so markers whose between-cluster differences are pure noise are not amplified."},
    {"key": "balanced_color_limit", "label": "Color limit", "type": "number", "min": 0.5, "max": 10.0, "step": 0.25,
     "help": "Symmetric clipping limit of the balanced color scale."},

    {"type": "header", "label": "High-resolution export",
     "help": "How the cluster partition is smoothed and vectorized."},
    {
        "key": "fine_export_smooth_sigma", "label": "Export smoothing sigma", "type": "number",
        "min": 0.0, "max": 20.0, "step": 0.25,
        "help": (
            "Winner-take-all Gaussian regularization on the clustering grid. Every valid-tissue pixel still "
            "receives exactly one class, so exported polygons share identical boundaries with no cracks."
        ),
    },
    {"key": "fine_export_max_sigma", "label": "Maximum smoothing sigma", "type": "number", "min": 0.0, "max": 30.0, "step": 0.5,
     "help": "Upper bound used when the export automatically increases smoothing to stay under the polygon budget."},
    {"key": "max_raw_polygons", "label": "Polygon budget", "type": "int", "min": 100, "max": 500_000, "step": 100,
     "help": "If vectorizing produces more polygons than this, smoothing is increased and the export retried."},
    {"key": "merged_valid_close_um", "label": "Valid-tissue closing (µm)", "type": "number", "min": 0.0, "max": 100.0, "step": 0.5,
     "help": "Closes narrow raster cracks in the merged valid-tissue polygon rebuilt from the retained clusters."},
    {"key": "merged_valid_fill_holes_um2", "label": "Valid-tissue hole fill (um^2)", "type": "number", "min": 0.0, "max": 100_000.0, "step": 10.0,
     "help": "Only holes smaller than this are filled, so genuinely removed lint regions stay excluded."},
]


# ==========================================================================
# Interactive heatmap plumbing
# ==========================================================================

def _domain_map() -> dict[int, str]:
    raw = st.session_state.get(DOMAIN_MAP_KEY, {}) or {}
    normalized: dict[int, str] = {}
    for cluster_id, domain in raw.items():
        try:
            normalized[int(cluster_id)] = str(domain)
        except (TypeError, ValueError):
            continue
    return normalized


def _set_domain_map(mapping: dict[int, str]) -> None:
    st.session_state[DOMAIN_MAP_KEY] = {int(key): str(value) for key, value in mapping.items()}


def _domain_names() -> list[str]:
    names = [str(name) for name in st.session_state.get(DOMAIN_NAMES_KEY, []) if str(name).strip()]
    for domain in _domain_map().values():
        if domain != UNASSIGNED and domain not in names:
            names.append(domain)
    return names


def _available_domain_options() -> list[str]:
    return [UNASSIGNED] + _domain_names()


def _build_merge_heatmap(
    balanced_scores: pd.DataFrame,
    shrunk_profiles: pd.DataFrame,
    cluster_order: list,
    marker_order: list,
    domain_map: dict[int, str],
    color_limit: float,
):
    """Plotly twin of the size-aware balanced clustermap, clusters on rows.

    The layout deliberately mirrors the seaborn clustermap above it: clusters on
    the Y axis in dendrogram order, markers on the X axis, the same diverging
    scale and the same printed shrunk mean Z-scores. Because clusters are rows,
    a selected or merged cluster is highlighted as a horizontal band.
    """
    import plotly.graph_objects as go

    cluster_order = [cluster for cluster in cluster_order if cluster in balanced_scores.index]
    marker_order = [marker for marker in marker_order if marker in balanced_scores.columns]
    color_matrix = balanced_scores.loc[cluster_order, marker_order].to_numpy()
    value_matrix = shrunk_profiles.loc[cluster_order, marker_order].to_numpy()
    row_labels = [f"Cluster {int(cluster)}" for cluster in cluster_order]
    row_labels = [
        f"{label}  ->  {domain_map.get(int(cluster), UNASSIGNED)}"
        if domain_map.get(int(cluster), UNASSIGNED) != UNASSIGNED else label
        for label, cluster in zip(row_labels, cluster_order)
    ]

    # The click selector reads cluster ids out of customdata: one id per cell,
    # repeated across each row so any cell in a row identifies its cluster.
    cluster_ids = np.tile(
        np.array([int(cluster) for cluster in cluster_order], dtype=object).reshape(-1, 1),
        (1, len(marker_order)),
    )

    show_values = len(cluster_order) * len(marker_order) <= 1600
    figure = go.Figure(
        data=go.Heatmap(
            z=color_matrix,
            x=marker_order,
            y=row_labels,
            customdata=cluster_ids,
            text=value_matrix,
            texttemplate="%{text:.2f}" if show_values else None,
            textfont=dict(size=9),
            colorscale="RdBu_r",
            zmid=0.0,
            zmin=-color_limit,
            zmax=color_limit,
            xgap=1,
            ygap=1,
            colorbar=dict(title="Balanced<br>score", thickness=14, len=0.75),
            hovertemplate=(
                "Cluster %{customdata}<br>%{x}<br>"
                "Shrunk mean Z: %{text:.2f}<extra></extra>"
            ),
        )
    )

    shapes = []
    for position, cluster in enumerate(cluster_order):
        if domain_map.get(int(cluster), UNASSIGNED) != UNASSIGNED:
            shapes.append(dict(
                type="rect",
                x0=-0.5, x1=len(marker_order) - 0.5,
                y0=position - 0.5, y1=position + 0.5,
                line=dict(color="rgba(40, 167, 69, 0.95)", width=3),
                fillcolor="rgba(40, 167, 69, 0.08)",
            ))

    assigned = sum(1 for cluster in cluster_order if domain_map.get(int(cluster), UNASSIGNED) != UNASSIGNED)
    figure.update_layout(
        title=dict(
            text=f"Cluster size-aware balanced signatures (merged: {assigned}/{len(cluster_order)})",
            x=0.5, xanchor="center", font=dict(size=15),
        ),
        dragmode=False,
        shapes=shapes,
        # Transparent backgrounds let the component inherit the Streamlit theme
        # instead of painting a white plot area on a dark page.
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(420, 30 * len(cluster_order) + 220),
        # automargin lets Plotly reserve room for the full tick labels, which is
        # what stops the cluster names and marker names from being clipped.
        margin=dict(l=20, r=20, t=70, b=20, autoexpand=True),
        xaxis=dict(
            title="Marker", side="bottom", tickangle=-60, automargin=True,
            tickfont=dict(size=11), showgrid=False, zeroline=False, constrain="domain",
        ),
        yaxis=dict(
            title="Cluster", automargin=True, autorange="reversed",
            tickfont=dict(size=12), showgrid=False, zeroline=False,
        ),
    )
    return figure, cluster_order, marker_order


def _apply_click_event(event: dict | None, domain_map: dict[int, str]) -> tuple[dict[int, str], list[int], str] | None:
    """Apply one buffered heatmap assignment, ignoring replays of the same payload.

    Streamlit components keep their last value across reruns, so the same
    payload comes back on the rerun that redraws the green frames.  Applying it
    only when it would actually change the map makes that replay harmless.
    """
    if not isinstance(event, dict) or event.get("action") != "apply":
        return None
    clusters: list[int] = []
    for value in event.get("selectedClusters", []) or []:
        try:
            clusters.append(int(value))
        except (TypeError, ValueError):
            continue
    domain = str(event.get("phenotype", "")).strip()
    if not clusters or not domain:
        return None
    if all(domain_map.get(cluster) == domain for cluster in clusters):
        return None
    updated = dict(domain_map)
    for cluster in clusters:
        updated[cluster] = domain
    return updated, sorted(set(clusters)), domain


# ==========================================================================
# Pipeline execution
# ==========================================================================

def _valid_tissue_candidates(sample_id: str) -> list[str]:
    directory = st.session_state.get("annot_ta_output_dir", "")
    if not directory or not os.path.isdir(directory):
        return []
    names = sorted(name for name in os.listdir(directory) if name.lower().endswith(".geojson"))
    # This step's own output and step 4's are partitions of the annotation, not
    # inputs to it. Offering them invites a run that clusters the labels of the
    # previous run instead of the tissue.
    names = [
        name for name in names
        if not name.endswith(("_macro_domains.geojson", "_zones.geojson", "_raw_macro_clusters.geojson"))
    ]
    preferred = [
        f"{sample_id}_valid_tissue.geojson",
        f"{sample_id}_valid_tissue_from_clusters_highres.geojson",
    ]
    ordered = [name for name in preferred if name in names]
    ordered += [name for name in names if name not in ordered and name.startswith(sample_id)]
    ordered += [name for name in names if name not in ordered]
    return ordered


def _run_clustering(sample_id: str, config: dict, marker_names: list[str], valid_tissue_file: str, status=None) -> dict:
    """Build the feature stack, cluster it, and characterise the clusters."""
    from spacemap.tissue_annotation_utils import (
        build_cluster_profiles,
        build_feature_stack,
        load_geojson_mask,
        run_pixel_clustering,
    )

    image_path = resolve_image_path(sample_id)
    if not image_path:
        raise FileNotFoundError(f"No image file found for sample '{sample_id}'.")
    reader = open_reader(image_path)
    marker_indices = [reader.channel_names.index(name) for name in marker_names if name in reader.channel_names]
    if not marker_indices:
        raise ValueError("Select at least one marker channel for clustering.")

    factor = int(config.get("clustering_downsample", 4))
    height, width = reader.target_shape(factor)

    if not os.path.isfile(valid_tissue_file):
        raise FileNotFoundError(f"Valid tissue GeoJSON not found: {valid_tissue_file}")
    # Reloaded from disk on every run so QuPath refinements are always honoured.
    valid_mask = load_geojson_mask(valid_tissue_file, (height, width), float(factor))
    if not valid_mask.any():
        raise ValueError(f"'{os.path.basename(valid_tissue_file)}' rasterized to an empty mask at {factor}x.")

    def progress(done: int, total: int, name: str) -> None:
        if status is not None:
            status.write(f"{sample_id}: reading marker {done}/{total} ({name})")

    feature_stack = build_feature_stack(reader, marker_indices, valid_mask, config, progress=progress)
    if status is not None:
        status.write(f"{sample_id}: clustering {feature_stack['features'].shape[0]:,} valid-tissue pixels...")
    clustering = run_pixel_clustering(feature_stack, config)
    profiles = build_cluster_profiles(feature_stack, clustering, config)

    with reader.open() as tif:
        reference_image = reader.read_channel(tif, marker_indices[0], factor)

    return {
        "feature_stack": feature_stack,
        "clustering": clustering,
        "profiles": profiles,
        "reference_image": reference_image,
        "valid_tissue_file": valid_tissue_file,
        "image_path": image_path,
        "config": dict(config),
        # Cache key for every figure derived from this run.
        "run_token": uuid.uuid4().hex,
    }


def _run_export(sample_id: str, state: dict, config: dict) -> dict:
    """Export the gapless cluster partition and the merged domain GeoJSON."""
    from spacemap.tissue_annotation_utils import export_cluster_partition, export_domain_partition

    output_dir = st.session_state.get("annot_ta_output_dir", "")
    os.makedirs(output_dir, exist_ok=True)

    excluded = [int(value) for value in st.session_state.get(EXCLUDED_KEY, [])]
    cluster_export = export_cluster_partition(
        state["clustering"], state["feature_stack"], config, output_dir, sample_id, excluded_clusters=excluded
    )

    domain_map = {
        cluster: domain
        for cluster, domain in _domain_map().items()
        if domain and domain != UNASSIGNED and cluster not in excluded
    }
    domain_export = None
    if domain_map:
        domain_export = export_domain_partition(
            cluster_export,
            domain_map,
            state["feature_stack"],
            state["clustering"],
            config,
            output_dir,
            sample_id,
            domain_colors=st.session_state.get(DOMAIN_COLORS_KEY, {}),
        )
        colors = dict(st.session_state.get(DOMAIN_COLORS_KEY, {}))
        colors.update(domain_export["domain_colors"])
        st.session_state[DOMAIN_COLORS_KEY] = colors
    return {
        "cluster_export": cluster_export,
        "domain_export": domain_export,
        "export_token": uuid.uuid4().hex,
    }


# ==========================================================================
# UI
# ==========================================================================

def _render_domain_manager() -> None:
    st.markdown("##### Domain labels")
    st.caption(
        "Define the tissue domains you want the clusters merged into, then click heatmap rows and apply a "
        "domain to all selected clusters at once. Labels are saved with the project, so they come back the "
        "next time you open it."
    )

    def _add_labels(new_labels: Sequence[str]) -> bool:
        """Append labels that are not already defined. Returns True if anything changed."""
        names = _domain_names()
        added = False
        for label in new_labels:
            label = str(label).strip()
            if label and label != UNASSIGNED and label not in names:
                names.append(label)
                added = True
        if added:
            st.session_state[DOMAIN_NAMES_KEY] = names
            save_project_settings()
        return added

    preset_column, preset_button_column = st.columns([4, 1])
    # Clearing the picker after adding has to happen before the widget is built.
    if st.session_state.pop("_annot_ta_s3_preset_choice_clear", False):
        st.session_state["annot_ta_s3_preset_choice"] = []
    preset_choice = preset_column.multiselect(
        "Common tissue labels",
        ALL_COMMON_DOMAIN_LABELS,
        default=[],
        key="annot_ta_s3_preset_choice",
        help=(
            "Curated starting catalogue grouped as "
            + "; ".join(f"{group}: {', '.join(labels)}" for group, labels in COMMON_DOMAIN_LABELS.items())
            + ". Pick the ones relevant to this tissue, or type your own below."
        ),
    )
    preset_button_column.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if preset_button_column.button("Add selected", key="annot_ta_s3_add_presets", width="stretch"):
        if _add_labels(preset_choice):
            st.session_state["_annot_ta_s3_preset_choice_clear"] = True
            st.rerun()

    entry_column, add_column = st.columns([4, 1])
    new_name = entry_column.text_input(
        "Custom domain name",
        value="",
        key="annot_ta_s3_new_domain",
        placeholder="e.g. Peritumoral_Stroma",
        help="Any label not in the catalogue. It is saved with the project alongside the preset ones.",
    )
    add_column.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if add_column.button("Add", key="annot_ta_s3_add_domain", width="stretch"):
        if _add_labels([new_name]):
            st.rerun()

    names = _domain_names()
    if names:
        from spacemap.tissue_annotation_utils import assign_distinct_colors

        stored_colors = dict(st.session_state.get(DOMAIN_COLORS_KEY, {}))
        # Defaults come from a maximally distinct palette rather than a hash, so
        # two domains never end up in near-identical hues. Colours the user
        # picked are kept and excluded from the pool.
        defaults = assign_distinct_colors(names, existing=stored_colors)
        st.caption("Colours are auto-assigned for maximum contrast. Override any of them here.")
        colors = {}
        color_columns = st.columns(min(4, len(names)))
        changed = False
        for index, name in enumerate(names):
            default_color = defaults[name]
            with color_columns[index % len(color_columns)]:
                picked = st.color_picker(name, value=default_color, key=f"annot_ta_s3_color_{name}")
            if picked != stored_colors.get(name):
                changed = True
            colors[name] = picked
        st.session_state[DOMAIN_COLORS_KEY] = colors
        if changed:
            save_project_settings()

        remove = st.multiselect(
            "Remove domain labels", names, default=[], key="annot_ta_s3_remove_domains",
            help="Removing a label also resets any cluster currently assigned to it.",
        )
        if remove and st.button("Remove selected labels", key="annot_ta_s3_remove_btn"):
            st.session_state[DOMAIN_NAMES_KEY] = [name for name in names if name not in remove]
            _set_domain_map({
                cluster: (UNASSIGNED if domain in remove else domain)
                for cluster, domain in _domain_map().items()
            })
            save_project_settings()
            st.rerun()
    else:
        st.info("Add at least one domain label before merging clusters.")


def _render_merge_interface(sample_id: str, state: dict) -> None:
    from .heatmap_component import heatmap_click_selector_available, render_heatmap_click_selector

    profiles = state["profiles"]
    clustering = state["clustering"]
    run_token = state["run_token"]

    # The dendrogram ordering is needed on every rerun, but drawing the three
    # clustermaps is not: build them once per clustering run and cache the
    # rendered images.
    orders = state.get("signature_orders")
    if orders is None:
        signature_figures = figures.cluster_signature_figures(profiles)
        from .common import save_figure

        import matplotlib.pyplot as plt

        rendered = {}
        for key, name in (("absolute", "step3_cluster_absolute"), ("z_score", "step3_cluster_zscore"),
                          ("balanced", "step3_cluster_balanced")):
            from io import BytesIO

            buffer = BytesIO()
            signature_figures[key].savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            save_figure(signature_figures[key], sample_id, name)
            plt.close(signature_figures[key])
            rendered[key] = buffer.getvalue()
        orders = {
            "cluster_order": signature_figures["cluster_order"],
            "marker_order": signature_figures["marker_order"],
            "images": rendered,
        }
        state["signature_orders"] = orders

    with st.expander("Cluster signature heatmaps (all three views)", expanded=False):
        st.caption(
            "Absolute intensity shows what each cluster is bright in. Mean Z-score shows what is *enriched* "
            "relative to the rest of the tissue. The size-aware balanced view additionally shrinks unreliable "
            "small clusters and equalises contrast per marker; it is the one used for merging below."
        )
        for key in ("absolute", "z_score", "balanced"):
            st.image(orders["images"][key], width="stretch")
        st.caption(f"All three figures are saved under `{output_dir_for(sample_id, 'figures')}`.")

    st.markdown("##### Clusters on tissue")
    st.caption(
        "Spatial layout of the raw cluster labels, available immediately after clustering. Use it together "
        "with the heatmaps to decide which clusters belong to the same tissue domain."
    )
    show_cached_figure(
        sample_id,
        "step3_cluster_map",
        run_token,
        lambda: figures.cluster_map_figure(
            clustering["cluster_map"],
            clustering["cluster_colors"],
            profiles["size_summary"],
            sample_id,
            clustering.get("reserved_low_signal_cluster"),
        ),
    )

    _render_domain_manager()

    domain_map = _domain_map()
    cluster_ids = [int(value) for value in profiles["balanced_scores"].index]
    for cluster in cluster_ids:
        domain_map.setdefault(cluster, UNASSIGNED)
    _set_domain_map(domain_map)

    st.markdown("##### Merge clusters on the heatmap")
    st.caption(
        "Same layout as the size-aware balanced clustermap above: clusters are rows in dendrogram order, "
        "so similar signatures sit next to each other. Click any cell to buffer that cluster row - buffered "
        "rows are framed in amber - then choose a domain and apply. Rows already merged are framed in green "
        "and labelled with their domain."
    )

    figure, cluster_order, marker_order = _build_merge_heatmap(
        profiles["balanced_scores"],
        profiles["shrunk_profiles"],
        orders["cluster_order"],
        orders["marker_order"],
        domain_map,
        float(profiles["color_limit"]),
    )

    if heatmap_click_selector_available():
        marker_scores = {
            str(int(cluster)): {
                str(marker): float(profiles["shrunk_profiles"].loc[cluster, marker])
                for marker in marker_order
            }
            for cluster in cluster_order
        }
        event = render_heatmap_click_selector(
            figure=figure,
            available_phenotypes=_available_domain_options(),
            cluster_display_labels={str(int(cluster)): f"Cluster {int(cluster)}" for cluster in cluster_order},
            cluster_marker_scores=marker_scores,
            reset_counter=int(st.session_state.get("annot_ta_s3_heatmap_component_reset_counter", 0)),
            key="annot_ta_s3_heatmap_click_selector",
        )
        applied = _apply_click_event(event, domain_map)
        if applied is not None:
            updated_map, clusters, domain = applied
            _set_domain_map(updated_map)
            st.session_state["annot_ta_s3_heatmap_component_reset_counter"] = int(st.session_state.get("annot_ta_s3_heatmap_component_reset_counter", 0)) + 1
            save_project_settings()
            st.success(f"Merged {len(clusters)} cluster(s) into '{domain}': {', '.join(str(c) for c in clusters)}")
            st.rerun()
    else:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False},
                        key="annot_ta_s3_heatmap_fallback")
        st.warning(
            "The click-to-merge heatmap component is not built in this installation. Use the assignment table "
            "below instead; it produces exactly the same result."
        )

    # --- assignment table (buffered inside a form, so edits do not rerun) ---
    st.markdown("##### Cluster assignment table")
    size_summary = profiles["size_summary"]
    shrunk = profiles["shrunk_profiles"]
    rows = []
    for cluster in cluster_order:
        cluster_int = int(cluster)
        top_markers = shrunk.loc[cluster].nlargest(5)
        rows.append({
            "Cluster": cluster_int,
            "Pixels": int(size_summary.loc[cluster, "pixels"]) if cluster in size_summary.index else 0,
            "% tissue": round(float(size_summary.loc[cluster, "tissue_percent"]), 2) if cluster in size_summary.index else 0.0,
            "Reliability": round(float(size_summary.loc[cluster, "reliability"]), 2) if cluster in size_summary.index else 0.0,
            "Top markers (shrunk Z)": ", ".join(f"{marker} ({value:+.2f})" for marker, value in top_markers.items()),
            "Domain": domain_map.get(cluster_int, UNASSIGNED),
        })
    summary_df = pd.DataFrame(rows)

    assigned = sum(1 for value in domain_map.values() if value != UNASSIGNED)
    header_left, header_right = st.columns([3, 1])
    header_left.caption("Edit domains in the table and press Sync. Nothing recomputes until you press it.")
    header_right.metric("Merged", f"{assigned}/{len(cluster_order)}")

    with st.form("annot_ta_s3_assignment_form"):
        edited = st.data_editor(
            summary_df,
            column_config={
                "Cluster": st.column_config.NumberColumn("Cluster", disabled=True),
                "Pixels": st.column_config.NumberColumn("Pixels", disabled=True),
                "% tissue": st.column_config.NumberColumn("% tissue", disabled=True, format="%.2f"),
                "Reliability": st.column_config.NumberColumn(
                    "Reliability", disabled=True, format="%.2f",
                    help="Cluster size divided by size plus the shrinkage prior. Low values mean the signature is based on few pixels.",
                ),
                "Top markers (shrunk Z)": st.column_config.TextColumn("Top markers (shrunk Z)", disabled=True, width="large"),
                "Domain": st.column_config.SelectboxColumn(
                    "Domain", options=_available_domain_options(), required=True,
                    help="Biological domain this cluster is merged into.",
                ),
            },
            disabled=["Cluster", "Pixels", "% tissue", "Reliability", "Top markers (shrunk Z)"],
            hide_index=True,
            num_rows="fixed",
            key="annot_ta_s3_assignment_editor",
            width="stretch",
        )
        synced = st.form_submit_button("Sync assignments", type="primary", width="stretch")

    if synced:
        _set_domain_map({int(row["Cluster"]): str(row["Domain"]) for _, row in edited.iterrows()})
        save_project_settings()
        st.success("Cluster assignments synced.")
        st.rerun()

    saved_table = save_table(summary_df.set_index("Cluster"), sample_id, "step3_cluster_assignments")
    if saved_table:
        st.caption(f"Assignment table saved to `{saved_table}`")

    # Publish the exact evidence behind the visible heatmaps. The assistant
    # reads this small record instead of guessing from the broad Annotation
    # page name or asking the user to locate a table that this tab already has.
    profile_tables = {
        "absolute": save_table(profiles["absolute_profiles"], sample_id, "step3_cluster_absolute_values"),
        "z_score": save_table(profiles["z_profiles"], sample_id, "step3_cluster_zscore_values"),
        "shrunk_z": save_table(profiles["shrunk_profiles"], sample_id, "step3_cluster_shrunk_z_values"),
        "balanced": save_table(profiles["balanced_scores"], sample_id, "step3_cluster_balanced_values"),
        "sizes": save_table(profiles["size_summary"], sample_id, "step3_cluster_sizes"),
    }
    spatial_summary = state.get("ai_spatial_summary")
    if not isinstance(spatial_summary, pd.DataFrame):
        spatial_summary = _cluster_spatial_summary(clustering["cluster_map"], cluster_order)
        state["ai_spatial_summary"] = spatial_summary
    profile_tables["spatial"] = save_table(
        spatial_summary.set_index("Cluster") if not spatial_summary.empty else spatial_summary,
        sample_id,
        "step3_cluster_spatial_summary",
    )
    figure_dir = Path(output_dir_for(sample_id, "figures"))
    evidence = st.session_state.setdefault("_spacemap_ai_page_evidence", {})
    evidence["tab4"] = {
        "section": "pixel_classification",
        "sample_id": sample_id,
        "image_path": str(state.get("image_path") or ""),
        "valid_tissue_path": str(state.get("valid_tissue_file") or ""),
        "cluster_count": len(cluster_order),
        "marker_count": len(state["feature_stack"].get("marker_names") or []),
        "markers": list(state["feature_stack"].get("marker_names") or []),
        "assignment_csv": str(saved_table or ""),
        "profile_tables": {key: str(value or "") for key, value in profile_tables.items()},
        "figures": {
            "absolute_heatmap": str(figure_dir / "step3_cluster_absolute.png"),
            "zscore_heatmap": str(figure_dir / "step3_cluster_zscore.png"),
            "balanced_heatmap": str(figure_dir / "step3_cluster_balanced.png"),
            "cluster_map": str(figure_dir / "step3_cluster_map.png"),
        },
        "current_domain_assignments": {
            str(cluster): str(domain_map.get(int(cluster), UNASSIGNED)) for cluster in cluster_order
        },
    }

    merged_assignments = {
        cluster: domain for cluster, domain in domain_map.items() if domain != UNASSIGNED
    }
    if merged_assignments:
        st.markdown("##### Merge preview")
        st.caption(
            "Spatial preview of the current assignments, rebuilt from the cluster map without vectorizing. "
            "It is drawn on request so that assigning domains stays instant; press the button again after "
            "changing assignments to refresh it."
        )
        colors = st.session_state.get(DOMAIN_COLORS_KEY, {})
        signature = json.dumps(
            [sorted(merged_assignments.items()), sorted(colors.items()), run_token],
            sort_keys=True, default=str,
        )
        preview_key = "annot_ta_s3_preview_signature"
        if st.button("Draw merge preview", key="annot_ta_s3_preview_btn", width="stretch"):
            st.session_state[preview_key] = signature
        requested = st.session_state.get(preview_key)
        if requested == signature:
            show_cached_figure(
                sample_id,
                "step3_merge_preview",
                signature,
                lambda: figures.domain_preview_figure(
                    clustering["cluster_map"], merged_assignments, colors, sample_id
                ),
            )
        elif requested:
            st.caption("Assignments changed since the last preview. Press the button to redraw it.")

    # A previous run may have had more clusters; Streamlit rejects stored
    # multiselect values that are no longer options.
    stored_excluded = st.session_state.get(EXCLUDED_KEY)
    if isinstance(stored_excluded, list) and any(value not in cluster_order for value in stored_excluded):
        st.session_state[EXCLUDED_KEY] = [value for value in stored_excluded if value in cluster_order]
    st.multiselect(
        "Clusters to exclude as residual lint",
        cluster_order,
        default=[value for value in st.session_state.get(EXCLUDED_KEY, []) if value in cluster_order],
        key=EXCLUDED_KEY,
        help=(
            "Excluded clusters are dropped from the merged valid-tissue polygon and from the domain export. "
            "Use this for clusters that turn out to be residual lint or debris."
        ),
    )


def render(project_root: Path, sample_id: str) -> None:
    st.markdown("### Step 3 - Pixel classification of valid tissue")
    st.caption(
        "Normalizes the marker stack over the reviewed valid tissue, clusters the pixels, and merges those "
        "clusters into biological domains. This step works on one image at a time because the cluster "
        "identities and the merge decisions are image-specific."
    )


    config = st.session_state.get(CONFIG_KEY, {})
    image_path = resolve_image_path(sample_id)
    if not image_path:
        st.warning("Selected image file could not be resolved.")
        return
    reader = open_reader(image_path)

    left, right = st.columns([1.35, 1.0])

    with left:
        st.markdown("#### Valid tissue source")
        candidates = _valid_tissue_candidates(sample_id)
        if not candidates:
            st.warning(
                f"No GeoJSON files found in `{st.session_state.get('annot_ta_output_dir', '')}`. Run steps 1 and 2 first."
            )
            return
        valid_file = sticky_file_selectbox(
            label="Valid tissue GeoJSON",
            candidates=candidates,
            preferred=f"{sample_id}_valid_tissue.geojson",
            key=f"annot_ta_s3_valid_file_{sample_id}",
            help_text=(
                "The annotation steps 1, 2 and 2b have been building. Pick a different file only if you "
                "saved a refinement under another name in QuPath. It is re-read from disk on every run, "
                "never from cache."
            ),
        )
        valid_tissue_path = os.path.join(st.session_state.get("annot_ta_output_dir", ""), valid_file)

        st.markdown("#### Marker channels")
        from spacemap.tissue_annotation_utils import split_he_and_codex_channels

        he_indices, codex_indices = split_he_and_codex_channels(reader.channel_names)
        all_names = list(reader.channel_names)
        codex_defaults = [all_names[i] for i in codex_indices]
        saved_markers = [name for name in st.session_state.get(MARKERS_KEY, []) if name in all_names]
        if "_annot_ta_s3_markers_pending" in st.session_state:
            st.session_state["annot_ta_s3_markers"] = st.session_state.pop("_annot_ta_s3_markers_pending")
        widget_value = st.session_state.get("annot_ta_s3_markers")
        if isinstance(widget_value, list) and any(name not in all_names for name in widget_value):
            st.session_state["annot_ta_s3_markers"] = [name for name in widget_value if name in all_names]
        selected_markers = st.multiselect(
            "Markers used for clustering",
            all_names,
            default=saved_markers or codex_defaults,
            key="annot_ta_s3_markers",
            help=(
                "Feature space for the pixel clustering. The three virtual H&E layers are excluded by default. "
                "Dropping uninformative markers usually produces cleaner, more interpretable domains."
            ),
        )
        marker_columns = st.columns(3)
        if marker_columns[0].button("All CODEX", key="annot_ta_s3_markers_codex", width="stretch"):
            # The widget key wins over `default` once it exists, so a preset has
            # to write the widget key as well as the canonical saved value.
            st.session_state[MARKERS_KEY] = codex_defaults
            st.session_state["_annot_ta_s3_markers_pending"] = codex_defaults
            st.rerun()
        if marker_columns[1].button("All channels", key="annot_ta_s3_markers_all", width="stretch"):
            st.session_state[MARKERS_KEY] = all_names
            st.session_state["_annot_ta_s3_markers_pending"] = all_names
            st.rerun()
        if marker_columns[2].button("Save selection", key="annot_ta_s3_markers_save", width="stretch"):
            st.session_state[MARKERS_KEY] = selected_markers
            save_project_settings()
            st.success("Marker selection saved.")

        with st.expander("Parameters", expanded=False):
            updated_config, submitted = render_parameter_form(
                form_key="annot_ta_s3_form",
                spec=PARAMETER_SPEC,
                config=config,
                submit_label="Save clustering settings",
            )
            if submitted:
                st.session_state[MARKERS_KEY] = selected_markers
                save_config(CONFIG_KEY, updated_config)
                st.success("Clustering settings saved to the project configuration.")
                config = updated_config

        estimated_height, estimated_width = reader.target_shape(int(config.get("clustering_downsample", 4)))
        st.caption(
            f"Clustering grid: {estimated_height} x {estimated_width} px "
            f"({estimated_height * estimated_width / 1e6:.1f} Mpx) x {len(selected_markers)} markers "
            f"~ {estimated_height * estimated_width * max(len(selected_markers), 1) * 4 / 1e9:.1f} GB feature stack. "
            "Increase the clustering downsample if this is too large for the machine."
        )

        if st.button("Run clustering", type="primary", width="stretch", key="annot_ta_s3_run"):
            status = st.empty()
            started = time.time()
            record_process_event(
                TAB_CONTEXT, "Pixel clustering started",
                details={"sample": sample_id, "markers": len(selected_markers), "k": config.get("n_clusters")},
            )
            try:
                with st.spinner("Building feature stack and clustering..."):
                    state = _run_clustering(sample_id, config, selected_markers, valid_tissue_path, status=status)
                store = runtime_store(sample_id)
                store["classification"] = state
                store.pop("exports", None)
                log_annotation(
                    f"{sample_id}: clustered {state['feature_stack']['features'].shape[0]:,} pixels into "
                    f"{state['clustering']['n_clusters']} clusters (Top-K={state['feature_stack']['top_k_effective']})",
                    subtab="s3",
                )
                record_process_event(
                    TAB_CONTEXT, "Pixel clustering complete", status="complete",
                    duration_s=time.time() - started,
                    details={"sample": sample_id, "clusters": state["clustering"]["n_clusters"]},
                )
                status.empty()
                st.success("Clustering complete.")
            except Exception as error:
                status.empty()
                st.error(f"Clustering failed: {error}")
                log_annotation(f"{sample_id}: clustering failed - {error}", subtab="s3")

        state = runtime_store(sample_id).get("classification")
        if state is None:
            st.info("Run clustering to build the cluster signatures and the merge heatmap.")
            return

        feature_stack = state["feature_stack"]
        metrics = st.columns(4)
        metrics[0].metric("Valid tissue pixels", f"{feature_stack['features'].shape[0]:,}")
        metrics[1].metric("Markers", f"{len(feature_stack['marker_names'])}")
        metrics[2].metric("Top-K used", f"{feature_stack['top_k_effective']}")
        metrics[3].metric("Clusters", f"{state['clustering']['n_clusters']}")
        st.caption(f"Automatic Top-K estimate for this sample and panel: {feature_stack['top_k_auto']}")

        with st.expander("Normalization diagnostics", expanded=False):
            run_token = state["run_token"]
            show_cached_figure(
                sample_id, "step3_valid_tissue_sync", run_token,
                lambda: figures.valid_tissue_sync_figure(
                    state["reference_image"], feature_stack["valid_tissue_mask"],
                    feature_stack["downsample"], sample_id,
                ),
            )
            show_cached_figure(
                sample_id, "step3_intensity_distributions", run_token,
                lambda: figures.intensity_distribution_figure(feature_stack, sample_id),
            )
            show_cached_figure(
                sample_id, "step3_topk_elbow", run_token,
                lambda: figures.top_k_elbow_figure(feature_stack, state["config"], sample_id),
            )
            cofactors = pd.DataFrame(
                sorted(feature_stack["cofactors"].items()), columns=["Marker", "Arcsinh cofactor"]
            )
            st.dataframe(cofactors, hide_index=True, width="stretch")
            save_table(cofactors.set_index("Marker"), sample_id, "step3_arcsinh_cofactors")

        _render_merge_interface(sample_id, state)

        st.markdown("#### Export")
        st.caption(
            "Writes the gapless per-cluster GeoJSON, the rebuilt valid-tissue polygon and, when at least one "
            "domain is assigned, the merged domain GeoJSON used by step 4. Exporting with no domains assigned "
            "is a valid workflow: open the per-cluster file in QuPath, merge and re-class the clusters there, "
            "and step 4 will pick up whichever file you point it at."
        )
        if st.button("Export cluster and domain GeoJSON", type="primary", width="stretch", key="annot_ta_s3_export"):
            started = time.time()
            try:
                with st.spinner("Vectorizing the partition..."):
                    exports = _run_export(sample_id, state, state["config"])
                runtime_store(sample_id)["exports"] = exports
                cluster_export = exports["cluster_export"]
                log_annotation(
                    f"{sample_id}: exported {cluster_export['polygon_count']:,} polygons at "
                    f"sigma={cluster_export['smoothing_sigma']:.2f}",
                    subtab="s3",
                )
                record_process_event(
                    TAB_CONTEXT, "Cluster export complete", status="complete",
                    duration_s=time.time() - started,
                    details={
                        "sample": sample_id,
                        "polygons": cluster_export["polygon_count"],
                        "excluded_clusters": cluster_export["excluded_clusters"],
                    },
                )
                st.success("Export complete.")
            except Exception as error:
                st.error(f"Export failed: {error}")
                log_annotation(f"{sample_id}: export failed - {error}", subtab="s3")

        exports = runtime_store(sample_id).get("exports")
        if exports:
            cluster_export = exports["cluster_export"]
            for message in cluster_export.get("retries", []):
                st.caption(message)
            st.caption(
                f"Exported at {cluster_export['export_resolution_um']:.3f} um/pixel with shared boundaries; "
                f"{cluster_export['polygon_count']:,} raw polygons at sigma={cluster_export['smoothing_sigma']:.2f}."
            )
            export_token = exports["export_token"]
            show_cached_figure(
                sample_id, "step3_cluster_partition", export_token,
                lambda: figures.cluster_partition_figure(
                    state["clustering"]["cluster_map"],
                    cluster_export["export_label_map"],
                    state["clustering"]["cluster_colors"],
                    f"fine gapless partition (sigma={cluster_export['smoothing_sigma']:.2f})",
                    sample_id,
                ),
            )

            domain_export = exports.get("domain_export")
            if domain_export is None:
                st.warning(
                    "No domains were assigned, so no merged domain GeoJSON was written. Merge the clusters "
                    "on the heatmap above, or merge them in QuPath using the per-cluster file below."
                )
            else:
                with st.expander("Merged domain signatures", expanded=False):
                    show_cached_figure(
                        sample_id, "step3_domain_absolute", export_token,
                        lambda: figures.domain_signature_figures(domain_export)["absolute"],
                    )
                    show_cached_figure(
                        sample_id, "step3_domain_zscore", export_token,
                        lambda: figures.domain_signature_figures(domain_export)["z_score"],
                    )
                show_cached_figure(
                    sample_id, "step3_domain_map", export_token,
                    lambda: figures.domain_map_figure(domain_export, sample_id),
                )

        # Output files are listed from disk, not from this session's run, so they
        # stay visible after a reload and can be taken straight into QuPath.
        st.markdown("##### Output files")
        describe_output(geojson_path(sample_id, "raw_macro_clusters"), "Per-cluster GeoJSON")
        describe_output(geojson_path(sample_id, "valid_tissue_from_clusters_highres"), "Rebuilt valid tissue GeoJSON")
        describe_output(geojson_path(sample_id, "macro_domains"), "Merged domain GeoJSON")

        clusters_on_disk = geojson_path(sample_id, "raw_macro_clusters")
        domains_on_disk = geojson_path(sample_id, "macro_domains")
        if os.path.isfile(domains_on_disk):
            qupath_hint(domains_on_disk, "Step 4")
        elif os.path.isfile(clusters_on_disk):
            st.info(
                f"**Review step:** open `{os.path.basename(clusters_on_disk)}` in QuPath, merge and re-class the "
                "clusters into tissue domains there, and save the result in the tissue annotation folder. "
                "Step 4 lets you select any GeoJSON in that folder, so a QuPath-merged file works exactly like "
                "the heatmap-merged one."
            )

    with right:
        st.markdown("#### Preview")
        domains_path = geojson_path(sample_id, "macro_domains")
        clusters_path = geojson_path(sample_id, "raw_macro_clusters")
        layer_options = {
            "Merged domains": domains_path,
            "All clusters": clusters_path,
            "Valid tissue": os.path.join(st.session_state.get("annot_ta_output_dir", ""), _valid_tissue_candidates(sample_id)[0])
            if _valid_tissue_candidates(sample_id) else "",
        }
        available = [name for name, path in layer_options.items() if path and os.path.isfile(path)]
        if available:
            prune_widget_selection("annot_ta_s3_layers", available)
            chosen = st.multiselect(
                "Overlay layers", available,
                default=["Merged domains"] if "Merged domains" in available else available[:1],
                key="annot_ta_s3_layers",
                help="Polygons are read from disk, so a QuPath refinement is reflected as soon as you save it.",
            )
            if chosen:
                image, colors = overlay_polygons_on_thumbnail(
                    image_path,
                    [layer_options[name] for name in chosen],
                    class_colors=st.session_state.get(DOMAIN_COLORS_KEY, {}),
                    fill_alpha=90,
                )
                st.image(image, width="stretch", caption=f"{sample_id} - {', '.join(chosen)}")
                render_class_legend(colors)
            else:
                render_image_preview(image_path, "annot_ta_s3")
        else:
            render_image_preview(image_path, "annot_ta_s3")

        st.caption(f"Figures and tables for this image: `{output_dir_for(sample_id, 'figures')}`")
