"""Single source of truth for assistant action coverage across SPACEMAP."""

from __future__ import annotations

from typing import Final


ACTION_CATALOG: Final[tuple[dict[str, str], ...]] = (
    {
        "page": "All pages",
        "operation": "Read SPACEMAP code and run a protected, reviewed scientific job",
        "status": "Ready",
    },
    {
        "page": "1. Registration",
        "operation": "Register one CODEX + H&E image pair",
        "status": "Ready",
    },
    {
        "page": "2. Background",
        "operation": "Blank-cycle subtraction and automatic background detection",
        "status": "Protected job",
    },
    {
        "page": "3. ROI",
        "operation": "Extract ROI images from checked GeoJSON boundaries",
        "status": "Ready",
    },
    {
        "page": "9. Visualization",
        "operation": (
            "Display named image channels, H&E, or a representative CODEX composite "
            "with optional segmentation/annotation GeoJSON overlays directly in Assistant chat"
        ),
        "status": "Ready",
    },
    {
        "page": "3. ROI",
        "operation": "Propose/draw TMA or free-tissue boundaries",
        "status": "Protected job",
    },
    {
        "page": "4. Annotation",
        "operation": "Tissue detection, artifact removal, blur parameter tests, pixel classification, zones",
        "status": "Ready",
    },
    {
        "page": "5. Segmentation",
        "operation": "Cellpose/DeepCell segmentation, feature extraction, spatial tagging",
        "status": "Protected job",
    },
    {
        "page": "6. Batch correction",
        "operation": "Filtering, correction, QC, and CSV export",
        "status": "Protected job",
    },
    {
        "page": "7. Phenotyping",
        "operation": "Leiden/SOM/Resolve/SCIMAP/Vortex and manual-reference tools",
        "status": "Protected job",
    },
    {
        "page": "8. Spatial analysis",
        "operation": "Density, neighborhoods, interactions, triads, and clonality",
        "status": "Protected job",
    },
    {
        "page": "9. Visualization",
        "operation": "Heatmaps, UMAP, overlays, and publication figures",
        "status": "Protected job",
    },
)


def ready_action_names() -> list[str]:
    """Return plain-language names for tools the assistant can execute now."""
    return [item["operation"] for item in ACTION_CATALOG if item["status"] == "Ready"]
