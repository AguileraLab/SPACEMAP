"""Figure builders for the Annotation tab.

The implementations live in :mod:`spacemap.tissue_annotation_figures` so the
tab and ``notebooks/Step4_annotation.ipynb`` render identical diagnostics from
one source. This module re-exports them under the name the subtabs use.
"""

from __future__ import annotations

from spacemap.tissue_annotation_figures import (  # noqa: F401
    PLOT_STYLE,
    artifact_figure,
    blur_diagnostic_figure,
    cluster_map_figure,
    cluster_partition_figure,
    cluster_signature_figures,
    domain_map_figure,
    domain_preview_figure,
    domain_signature_figures,
    intensity_distribution_figure,
    tissue_mask_figure,
    top_k_elbow_figure,
    valid_tissue_sync_figure,
)

__all__ = [
    "PLOT_STYLE",
    "artifact_figure",
    "blur_diagnostic_figure",
    "cluster_map_figure",
    "cluster_partition_figure",
    "cluster_signature_figures",
    "domain_map_figure",
    "domain_preview_figure",
    "domain_signature_figures",
    "intensity_distribution_figure",
    "tissue_mask_figure",
    "top_k_elbow_figure",
    "valid_tissue_sync_figure",
]
