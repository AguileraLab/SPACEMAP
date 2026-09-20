"""Small, safe context snapshots for the app-wide assistant."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .action_catalog import ready_action_names
from .paths import is_assistant_read_allowed


PAGE_TITLES = {
    "tab0": "Project Workspace",
    "tab1": "Registration",
    "tab2": "Background Subtraction",
    "tab3": "ROI Extraction",
    "tab4": "Annotation",
    "tab5": "Segmentation",
    "tab6": "Batch Correction",
    "tab7": "Phenotyping",
    "tab8": "Spatial Analysis",
    "tab9": "Visualization",
    "assistant": "AI Assistant",
}

PAGE_SECTIONS = {
    "tab0": ["Project selection", "Pipeline directories", "Project inventory", "Backup and portability"],
    "tab1": ["CODEX and H&E batch registration", "Multi-stack registration"],
    "tab2": ["Background subtraction", "Background detection"],
    "tab3": ["Image and boundary matching", "ROI extraction", "Channel export"],
    "tab4": ["Tissue detection", "Artifact removal", "Tissue pieces", "Pixel classification", "Pooling and zones"],
    "tab5": ["Input and annotation matching", "Cellpose segmentation", "Cell features", "Spatial tagging"],
    "tab6": ["File selection", "Quality filtration", "Batch correction", "QC and export"],
    "tab7": ["Leiden", "SOM", "Threshold", "SCIMAP", "Vortex", "Resolve clustering", "Manual Reference", "MR-ML", "Consensus ML", "Subphenotyping", "Visualization", "Tools"],
    "tab8": ["Feature screening", "Density", "Neighborhoods", "Cell interactions", "Triads", "Clonality", "ROI explorer", "Virtual tissue"],
    "tab9": ["Actual tissue", "Cell visualization", "Heatmaps", "Marker analysis", "Pairwise plots", "Proportions", "UMAP", "Validation", "Publication figures"],
}

STATE_PREFIXES = {
    "tab0": ("active_project", "proj_", "project_"),
    "tab1": ("tab1_", "batch_codex", "batch_he"),
    "tab2": ("tab2_", "bg_"),
    "tab3": ("tab3_",),
    "tab4": ("tab4_", "annot_", "cell_annot"),
    "tab5": ("tab4_", "seg_", "cellpose"),
    "tab6": ("bc_", "tab6_"),
    "tab7": ("global_pheno", "tab7_", "leiden_", "som_", "threshold_", "scimap_", "vortex_", "mrml_", "tools_"),
    "tab8": ("spatial_", "tab8_", "density_", "neighborhood", "interaction", "triad", "clonality"),
    "tab9": ("global_visualization", "tab9_", "visual_", "heatmap_", "umap_"),
}

SENSITIVE_KEY_PARTS = ("password", "secret", "token", "api_key", "credential")
MAX_SETTINGS = 36
MAX_STRING = 600
PAGE_EVIDENCE_KEY = "_spacemap_ai_page_evidence"

PRIORITY_STATE_KEYS = {
    "tab0": (
        "proj_mpp_codex", "proj_mpp_he", "proj_use_gpu", "proj_gpu_id",
        "proj_num_workers", "batch_codex_dir", "batch_he_dir", "tab1_batch_output_dir",
    ),
    "tab1": (
        "batch_codex_dir", "batch_he_dir", "tab1_batch_output_dir",
        "batch_mpp_codex", "batch_mpp_he", "batch_n_keypoints",
    ),
}


def _safe_scalar(value: Any) -> Any | None:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (str, Path)):
        text = str(value)
        return text[:MAX_STRING] + ("…" if len(text) > MAX_STRING else "")
    if isinstance(value, (list, tuple)) and len(value) <= 30:
        converted = [_safe_scalar(item) for item in value]
        if all(item is not None for item in converted):
            return converted
    return None


def _matching_settings(page_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = STATE_PREFIXES.get(page_id, ())
    settings: dict[str, Any] = {}
    for key in PRIORITY_STATE_KEYS.get(page_id, ()):
        if key not in state:
            continue
        value = _safe_scalar(state.get(key))
        if value is not None or state.get(key) is None:
            settings[key] = value
    for key in sorted(str(item) for item in state.keys()):
        if key in settings:
            continue
        lowered = key.lower()
        if any(part in lowered for part in SENSITIVE_KEY_PARTS):
            continue
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        value = _safe_scalar(state.get(key))
        if value is None and state.get(key) is not None:
            continue
        settings[key] = value
        if len(settings) >= MAX_SETTINGS:
            break
    return settings


def _page_evidence(page_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded JSON-safe record explicitly published by a rendered tab."""
    container = state.get(PAGE_EVIDENCE_KEY)
    raw = container.get(page_id) if isinstance(container, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    clean: dict[str, Any] = {}
    for key, value in list(raw.items())[:40]:
        if any(part in str(key).casefold() for part in SENSITIVE_KEY_PARTS):
            continue
        if isinstance(value, Mapping):
            child = {
                str(child_key): scalar
                for child_key, child_value in list(value.items())[:40]
                if (scalar := _safe_scalar(child_value)) is not None
            }
            clean[str(key)] = child
        else:
            scalar = _safe_scalar(value)
            if scalar is not None:
                clean[str(key)] = scalar
    return clean


def build_page_context(
    page_id: str,
    state: Mapping[str, Any],
    *,
    guide: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe snapshot; it never opens a selected project data file."""
    active_project_root = str(state.get("active_project_root") or "")
    project_inside_boundary = bool(
        active_project_root and is_assistant_read_allowed(active_project_root)
    )
    payload: dict[str, Any] = {
        "page_id": page_id,
        "page_title": PAGE_TITLES.get(page_id, page_id),
        "registered_sections": PAGE_SECTIONS.get(page_id, []),
        "active_project_name": _safe_scalar(state.get("active_project_name")),
        "active_project_root": active_project_root,
        "assistant_can_open_project_files": project_inside_boundary,
        "current_settings": _matching_settings(page_id, state),
        "assistant_mode": "planning_with_registered_project_actions",
        "executable_actions": ready_action_names(),
    }
    evidence = _page_evidence(page_id, state)
    if evidence:
        payload["current_tab_evidence"] = evidence
    if guide:
        payload["purpose"] = guide.get("purpose", "")
        payload["before_run"] = guide.get("before_run", "")
        payload["next_step"] = guide.get("next_step", "")
        payload["declared_inputs"] = [list(item) for item in guide.get("inputs", ())]
        payload["declared_outputs"] = [list(item) for item in guide.get("outputs", ())]
    return payload


def build_workspace_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Small cross-tab snapshot so Chat can understand work done on other pages."""
    pages: dict[str, Any] = {}
    for page_id, title in PAGE_TITLES.items():
        if page_id == "assistant":
            continue
        settings = _matching_settings(page_id, state)
        evidence = _page_evidence(page_id, state)
        if settings or evidence:
            pages[page_id] = {
                "title": title,
                "settings": dict(list(settings.items())[:24]),
            }
            if evidence:
                pages[page_id]["current_tab_evidence"] = evidence
    return {
        "active_project_name": _safe_scalar(state.get("active_project_name")),
        "active_project_root": _safe_scalar(state.get("active_project_root")),
        "loaded_pages": pages,
        "note": "This is current UI state from all loaded pages, not only the visible page.",
    }


def context_as_text(context: Mapping[str, Any]) -> str:
    return json.dumps(context, indent=2, ensure_ascii=False, default=str)
