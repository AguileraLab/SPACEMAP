"""Lightweight navigation metadata; opening Studio never imports pipeline tabs."""

import hashlib
from pathlib import Path

STEPS = (
    ("tab0", "Project", "project", "Set up or change the project you are working on."),
    ("tab1", "Registration", "registration", "Align the images from the same sample."),
    ("tab2", "Background", "bg_subtraction", "Remove background signal from your markers."),
    ("tab3", "Regions of interest", "roi", "Choose the areas you want to study."),
    ("tab4", "Annotation", "annotation", "Draw and review tissue regions."),
    ("tab5", "Segmentation", "segmentation", "Find cells and measure their markers."),
    ("tagging", "Cell tagging", "segmentation", "Tag each cell with the tissue annotation it sits in, its tissue piece, beds and distances."),
    ("tab6", "Batch correction", "batch_correction", "Combine and compare measurements across images."),
    ("tab7", "Phenotyping", "phenotyping", "Review cells and give them phenotype labels."),
    ("tab8", "Spatial analysis", "analysis", "Explore where cells are and how they relate to each other."),
    ("tab9", "Visualization", "visualization", "Review your results and make figures."),
)


def context(channel, base=""):
    from spacemap.ai_settings import is_enabled
    from spacemap_gui.viewer import registry

    if registry.session_view(channel, mark_seen=False) is None:
        raise ValueError("Open Studio from SPACEMAP to link it to your project.")
    project = registry.session_project(channel) or {}
    return {
        "project": Path(project["root"]).name if project.get("root") else "No project open",
        "has_project": bool(project.get("root")),
        # Tells two projects with the same name apart, without giving the page the folder.
        "project_id": hashlib.sha256(str(project["root"]).encode()).hexdigest()[:12] if project.get("root") else None,
        "step": project.get("tab") if project.get("tab") in {s[0] for s in STEPS} else "tab0",
        "steps": [
            {"id": key, "name": name, "description": description, "classic": base + "/" + path}
            for key, name, path, description in STEPS
        ],
        "assistant": base + ("/assistant" if is_enabled() else "/project"),
        "assistant_label": "AI assistant" if is_enabled() else "Set up AI",
    }
