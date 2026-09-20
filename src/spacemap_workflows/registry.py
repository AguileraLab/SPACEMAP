"""One simple catalog of the SPACEMAP operations an agent may choose."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from spacemap_assistant.code_index import SourceFunction, is_planner_callable, load_source_index

from .schemas import ResourceProfile, ToolSpec


QUERY_SYNONYMS = {
    "register": {"registration", "align"},
    "registration": {"register", "align"},
    "background": {"subtract", "subtraction", "blank"},
    "roi": {"region", "boundary", "geojson", "crop"},
    "blur": {"blurry", "focus", "artifact"},
    "blurry": {"blur", "focus", "artifact"},
    "segment": {"segmentation", "cellpose", "deepcell"},
    "segmentation": {"segment", "cellpose", "deepcell"},
    "phenotype": {"phenotyping", "cluster", "classification"},
    "phenotyping": {"phenotype", "cluster", "classification"},
    "cluster": {"clustering", "leiden", "som", "vortex"},
    "clustering": {"cluster", "leiden", "som", "vortex"},
    "quality": {"qc", "validation", "score", "metric"},
    "compare": {"comparison", "score", "evaluate"},
    "correct": {"correction", "corrected", "corrector"},
    "annotate": {"annotation"},
    "spatial": {"neighborhood", "interaction", "density"},
    "image": {"tiff", "preview", "overlay"},
}
SEARCH_STOP_WORDS = {
    "a",
    "all",
    "and",
    "do",
    "for",
    "from",
    "make",
    "of",
    "on",
    "run",
    "save",
    "the",
    "to",
    "use",
    "using",
    "with",
}
DOMAIN_ANCHORS = {
    "cellpose",
    "deepcell",
    "leiden",
    "scimap",
    "som",
    "vortex",
    "umap",
    "geojson",
    "dapi",
}
PAGE_QUERY_TERMS = (
    (
        "1. Registration",
        ("register", "registration", "align codex", "align h&e", "align he", "merged image"),
    ),
    (
        "2. Background",
        ("background", "subtract blank", "subtraction", "blank image"),
    ),
    (
        "3. ROI",
        ("roi", "region of interest", "bounding box", "crop image"),
    ),
    (
        "4. Annotation",
        (
            "annotation",
            "annotate",
            "artifact",
            "blurry",
            "blur",
            "blood clot",
            "lint",
            "pixel classification",
            "valid tissue",
            "tissue detection",
        ),
    ),
    (
        "5. Segmentation",
        ("segmentation", "segment cell", "cellpose", "deepcell", "cell mask"),
    ),
    (
        "6. Batch correction",
        ("batch correction", "correct batch", "combat"),
    ),
    (
        "7. Phenotyping",
        (
            "phenotype",
            "phenotyping",
            "leiden",
            "clustering",
            "cluster",
            "manual reference",
            "canonical marker",
            "cell type",
        ),
    ),
    (
        "8. Spatial analysis",
        (
            "spatial analysis",
            "neighborhood",
            "cell interaction",
            "triad",
            "clonality",
            "spatial density",
        ),
    ),
    (
        "9. Visualization",
        ("visualization", "display", "figure", "heatmap", "thumbnail", "preview", "overlay", "umap"),
    ),
)


_TOOLS = (
    ToolSpec(
        name="registration.codex_he",
        version="1",
        page="1. Registration",
        summary="Register one CODEX image with one H&E image.",
        callable_path="spacemap.registration:register_images",
        permission="artifact_write",
        status="ready",
        inputs={"codex_image": "TIFF/QPTIFF", "he_image": "TIFF/QPTIFF", "output_dir": "folder"},
        parameters={"codex_mpp": "number", "he_mpp": "number", "rotation": "90-degree step"},
        outputs={"registered_image": "pyramidal OME-TIFF", "qc": "PNG and JSON report"},
        qc=("registered image exists", "channels preserved", "alignment preview"),
        resources=ResourceProfile(cpu="high", ram="high", gpu="optional", disk="high", long_running=True),
        keywords=("register", "registration", "align", "codex", "h&e"),
    ),
    ToolSpec(
        name="roi.extract",
        version="1",
        page="3. ROI",
        summary="Extract ROI images from a registered image and GeoJSON boundaries.",
        callable_path="spacemap.registration:process_single_image_rois",
        permission="artifact_write",
        status="ready",
        inputs={"image": "registered TIFF", "boundaries": "GeoJSON", "output_dir": "folder"},
        parameters={"mpp": "number or image metadata", "channels": "channel names"},
        outputs={"rois": "pyramidal OME-TIFF files", "qc": "boundary PNG and JSON report"},
        qc=("ROI count matches boundaries", "every TIFF opens", "channels preserved"),
        resources=ResourceProfile(cpu="high", ram="high", disk="high", long_running=True),
        keywords=("roi", "crop", "boundaries", "geojson", "extract"),
    ),
    ToolSpec(
        name="annotation.blur_optimize",
        version="1",
        page="4. Annotation",
        summary="Test blur settings, compare candidates, and save a provisional polygon set for review.",
        callable_path="spacemap_workflows.annotation:run_blur_parameter_search",
        permission="artifact_write",
        status="ready",
        inputs={"image": "ROI OME-TIFF", "tissue_boundary": "GeoJSON", "project_config": "project JSON"},
        parameters={"maximum_trials": "1-20", "mpp": "microns per pixel"},
        outputs={"trials": "GeoJSON per trial", "comparison": "CSV", "candidate": "provisional GeoJSON"},
        qc=("each trial is versioned", "polygon evidence present", "source image is never overwritten"),
        resources=ResourceProfile(cpu="high", ram="very high", gpu="optional", disk="standard", long_running=True),
        keywords=("blur", "blurry", "out of focus", "artifact", "parameter search", "polygon"),
    ),
    ToolSpec(
        name="annotation.bright_optimize",
        version="1",
        page="4. Annotation",
        summary=(
            "Test bright-glare brightness and smoothness settings, compare candidates, and save "
            "a provisional polygon set with local H&E/CODEX review images."
        ),
        callable_path="spacemap_workflows.annotation:run_bright_parameter_search",
        permission="artifact_write",
        status="ready",
        inputs={"image": "ROI OME-TIFF", "tissue_boundary": "GeoJSON", "project_config": "project JSON"},
        parameters={"maximum_trials": "1-20", "mpp": "microns per pixel"},
        outputs={"trials": "GeoJSON per trial", "comparison": "CSV", "candidate": "provisional GeoJSON"},
        qc=("each trial is versioned", "candidate marker evidence is preserved", "visual review uses small local previews"),
        resources=ResourceProfile(cpu="high", ram="very high", gpu="optional", disk="standard", long_running=True),
        keywords=("bright", "glare", "debris", "smooth", "artifact", "parameter search", "polygon"),
    ),
    ToolSpec(
        name="visualization.render",
        version="1",
        page="9. Visualization",
        summary="Display named channels with optional GeoJSON overlays and a scale bar.",
        callable_path="spacemap_assistant.preview_action:generate_marker_previews",
        permission="temporary_write",
        status="ready",
        inputs={"image": "OME-TIFF", "overlays": "zero or more GeoJSON files"},
        parameters={"channels": "channel names", "scale_bar_um": "number"},
        outputs={"preview": "PNG"},
        qc=("requested channels exist", "overlay coordinates match full-resolution image"),
        resources=ResourceProfile(),
        keywords=("display", "thumbnail", "preview", "overlay", "channel", "marker"),
    ),
    ToolSpec(
        name="spacemap.protected_backend",
        version="1",
        page="All pages",
        summary=(
            "Call a public SPACEMAP backend function inside the protected worker when no friendly "
            "tool adapter exists yet. Source stays read-only and outputs are checked."
        ),
        callable_path="spacemap_assistant.general_worker:run_general_action",
        permission="artifact_write",
        status="protected_backend",
        inputs={"inputs": "named existing paths", "function": "indexed public SPACEMAP callable"},
        parameters={"arguments": "JSON-safe values", "maximum_runtime": "1 minute to 48 hours"},
        outputs={"artifacts": "files inside a named project output folder", "report": "JSON"},
        qc=("callable exists", "code validation", "isolated staging", "output manifest"),
        resources=ResourceProfile(cpu="variable", ram="variable", gpu="optional", disk="variable", long_running=True),
        keywords=("background", "segment", "phenotype", "leiden", "cluster", "spatial", "figure", "batch"),
    ),
)


@lru_cache(maxsize=1)
def _friendly_catalog() -> dict[str, ToolSpec]:
    result: dict[str, ToolSpec] = {}
    for item in _TOOLS:
        result[item.validate().name] = item
    return result


def _page_for_module(module: str) -> str:
    lowered = module.casefold()
    tab_match = re.search(r"spacemap_gui\.tabs\.tab([1-9])", lowered)
    if tab_match:
        return {
            "1": "1. Registration",
            "2": "2. Background",
            "3": "3. ROI",
            "4": "4. Annotation",
            "5": "5. Segmentation",
            "6": "6. Batch correction",
            "7": "7. Phenotyping",
            "8": "8. Spatial analysis",
            "9": "9. Visualization",
        }[tab_match.group(1)]
    mappings = (
        ("registration", "1. Registration"),
        ("background", "2. Background"),
        ("roi", "3. ROI"),
        ("annotation", "4. Annotation"),
        ("tissue", "4. Annotation"),
        ("segmentation", "5. Segmentation"),
        ("batch", "6. Batch correction"),
        ("ph_", "7. Phenotyping"),
        ("phenotyp", "7. Phenotyping"),
        ("clustering", "7. Phenotyping"),
        ("spatial", "8. Spatial analysis"),
        ("tab8", "8. Spatial analysis"),
        ("visual", "9. Visualization"),
        ("figure", "9. Visualization"),
        ("heatmap", "9. Visualization"),
        ("umap", "9. Visualization"),
        ("tab9", "9. Visualization"),
    )
    return next((page for token, page in mappings if token in lowered), "Shared SPACEMAP backend")


def _resource_profile(record: SourceFunction) -> ResourceProfile:
    text = f"{record.module} {record.qualname} {record.signature} {record.docstring}".casefold()
    gpu = "optional" if any(word in text for word in ("gpu", "cellpose", "deepcell", "cuda")) else "none"
    heavy = any(
        word in text
        for word in (
            "batch",
            "cluster",
            "segmentation",
            "tiff",
            "image",
            "training",
            "validation",
        )
    )
    return ResourceProfile(
        cpu="high" if heavy else "standard",
        ram="high" if heavy else "standard",
        gpu=gpu,
        disk="high" if any(word in text for word in ("tiff", "image", "csv", "export")) else "standard",
        long_running=heavy,
    )


def _registered_name(record: SourceFunction) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{record.module}.{record.qualname}")


def _is_agent_callable(record: SourceFunction) -> bool:
    """Register public top-level functions; nested callbacks are implementation details."""
    return is_planner_callable(record)


@lru_cache(maxsize=1)
def registered_backend_tools() -> list[ToolSpec]:
    """Return the executable catalog generated from the current SPACEMAP source."""
    payload = load_source_index()
    tools: list[ToolSpec] = []
    for item in payload.get("functions", []):
        try:
            record = SourceFunction(**item)
        except TypeError:
            continue
        if not _is_agent_callable(record):
            continue
        summary = (record.docstring.strip().splitlines() or [record.signature])[0]
        keywords = tuple(sorted(_tokens(f"{record.module} {record.qualname} {record.signature}")))
        tools.append(
            ToolSpec(
                name=_registered_name(record),
                version="source-index-1",
                page=_page_for_module(record.module),
                summary=summary[:500] or record.signature,
                callable_path=record.callable_name,
                permission="artifact_write",
                status="protected_backend",
                inputs={"signature": record.signature},
                parameters={"arguments": "Use the exact indexed function signature."},
                outputs={"artifacts": "Files saved under the checked OUTPUT_DIR."},
                qc=("function comes from the current source index", "outputs are non-empty and logged"),
                resources=_resource_profile(record),
                keywords=keywords,
            ).validate()
        )
    return sorted(tools, key=lambda item: item.name)


@lru_cache(maxsize=1)
def _catalog() -> dict[str, ToolSpec]:
    result = dict(_friendly_catalog())
    for item in registered_backend_tools():
        result.setdefault(item.name, item)
    return result


def list_tools(*, include_manual: bool = False) -> list[ToolSpec]:
    values = list(_catalog().values())
    if not include_manual:
        values = [item for item in values if item.status != "manual_only"]
    return sorted(values, key=lambda item: item.name)


def tool_counts() -> dict[str, Any]:
    tools = list_tools()
    pages: dict[str, int] = {}
    for item in tools:
        pages[item.page] = pages.get(item.page, 0) + 1
    return {
        "total": len(tools),
        "friendly": len(_friendly_catalog()),
        "protected_backend": sum(item.status == "protected_backend" for item in tools),
        "pages": dict(sorted(pages.items())),
    }


def refresh_tool_registry() -> dict[str, Any]:
    """Forget in-process caches; the next read rebuilds from the source index."""
    registered_backend_tools.cache_clear()
    _catalog.cache_clear()
    return tool_counts()


def get_tool(name: str) -> ToolSpec:
    try:
        return _catalog()[str(name)]
    except KeyError as error:
        raise KeyError(f"Unknown SPACEMAP tool: {name}") from error


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value))
    expanded = expanded.replace("_", " ").replace(".", " ").replace("-", " ")
    raw = {item.casefold() for item in re.findall(r"[A-Za-z][A-Za-z0-9]+", expanded)}
    tokens = {item for item in raw if item not in SEARCH_STOP_WORDS}
    tokens |= {item[:-1] for item in tokens if len(item) > 4 and item.endswith("s") and not item.endswith("ss")}
    inflections = {
        "batches": "batch",
        "cells": "cell",
        "classes": "class",
        "rois": "roi",
    }
    tokens |= {singular for plural, singular in inflections.items() if plural in tokens}
    return tokens


def requested_pages(query: str) -> list[str]:
    """Return every pipeline area explicitly named in a plain-language request."""
    text = re.sub(r"\s+", " ", str(query).replace("_", " ").casefold())
    pages: list[str] = []
    for page, phrases in PAGE_QUERY_TERMS:
        if any(phrase in text for phrase in phrases):
            pages.append(page)
    return pages


def relevant_tools(query: str, *, limit: int = 8) -> list[ToolSpec]:
    wanted = _tokens(query)
    requested_anchors = wanted & DOMAIN_ANCHORS
    wanted |= {
        synonym
        for token in tuple(wanted)
        for synonym in QUERY_SYNONYMS.get(token, set())
    }
    scored: list[tuple[float, ToolSpec]] = []
    for tool in list_tools():
        callable_symbol = tool.callable_path.rsplit(":", 1)[-1]
        callable_tokens = _tokens(callable_symbol)
        name_tokens = _tokens(f"{tool.name} {tool.callable_path} {tool.page}")
        detail_tokens = _tokens(" ".join((tool.summary, *tool.keywords)))
        score = (
            8.0 * len(wanted & callable_tokens)
            + 4.0 * len(wanted & name_tokens)
            + len(wanted & detail_tokens)
        )
        score += 60.0 * len(requested_anchors & callable_tokens)
        score += 20.0 * len(requested_anchors & (name_tokens - callable_tokens))
        function_name = callable_symbol.replace("_", " ").casefold()
        if function_name and function_name in str(query).replace("_", " ").casefold():
            score += 20.0
        if tool.status == "ready" and score:
            score += 20.0
        if score:
            scored.append((score, tool))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    maximum = max(1, int(limit))
    pages = requested_pages(query)
    if len(pages) < 2:
        return [item for _score, item in scored[:maximum]]

    # Do not let one module fill every slot in a long multi-stage request.
    # Take one result from each named page, then keep cycling through those
    # pages before using the remaining global results.
    buckets = {
        page: [item for item in scored if item[1].page == page]
        for page in pages
    }
    chosen: list[tuple[float, ToolSpec]] = []
    chosen_names: set[str] = set()
    while len(chosen) < maximum:
        added = False
        for page in pages:
            bucket = buckets[page]
            while bucket and bucket[0][1].name in chosen_names:
                bucket.pop(0)
            if not bucket:
                continue
            item = bucket.pop(0)
            chosen.append(item)
            chosen_names.add(item[1].name)
            added = True
            if len(chosen) >= maximum:
                break
        if not added:
            break
    for item in scored:
        if len(chosen) >= maximum:
            break
        if item[1].name not in chosen_names:
            chosen.append(item)
            chosen_names.add(item[1].name)
    return [item for _score, item in chosen]


def tool_manifest(query: str = "", *, limit: int = 12) -> str:
    tools = relevant_tools(query, limit=limit) if query.strip() else list_tools()
    payload: list[dict[str, Any]] = [item.as_dict() for item in tools]
    return json.dumps(payload, ensure_ascii=False, indent=2)
