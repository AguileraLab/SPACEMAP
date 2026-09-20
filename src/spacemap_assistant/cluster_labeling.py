"""Small, tab-aware evidence package for pixel-cluster naming."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping

from .paths import require_assistant_read_path


PAGE_EVIDENCE_KEY = "_spacemap_ai_page_evidence"


def is_cluster_labeling_request(prompt: str) -> bool:
    text = str(prompt).casefold()
    cluster = bool(re.search(r"\bclusters?\b", text))
    expression = any(term in text for term in ("marker", "expression", "heatmap", "pixel classification"))
    naming = any(term in text for term in ("label", "name", "identify", "annotate", "phenotype", "what are"))
    return cluster and expression and naming


def _prompt_csv(prompt: str) -> Path | None:
    for raw in re.findall(r"(?<![A-Za-z0-9])(/[^\s`\"']+)", str(prompt).replace("\\_", "_")):
        path = Path(raw.rstrip(".,;:?!)]}"))
        if path.suffix.casefold() == ".csv":
            return path.resolve(strict=False)
    return None


def _read_csv(path: Path, limit: int = 250) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        return [dict(row) for _, row in zip(range(limit), csv.DictReader(stream))]


def _cluster_column(row: Mapping[str, Any]) -> str:
    for name in ("Cluster", "cluster", "Unnamed: 0", "index", ""):
        if name in row:
            return str(row.get(name) or "")
    return ""


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_summary(profile_path: Path, assignments: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows = _read_csv(profile_path)
    result: list[dict[str, Any]] = []
    ignored = {"Cluster", "cluster", "Unnamed: 0", "index", ""}
    for row in rows:
        cluster = _cluster_column(row)
        scores = [(name, value) for name, raw in row.items() if name not in ignored and (value := _float(raw)) is not None]
        scores.sort(key=lambda item: item[1], reverse=True)
        assignment = assignments.get(cluster, {})
        result.append(
            {
                "cluster": cluster,
                "pixels": assignment.get("Pixels", ""),
                "tissue_percent": assignment.get("% tissue", ""),
                "reliability": assignment.get("Reliability", ""),
                "current_domain": assignment.get("Domain", "Unassigned"),
                "enriched": ", ".join(f"{name} {value:+.2f}" for name, value in scores[:8]),
                "depleted": ", ".join(f"{name} {value:+.2f}" for name, value in scores[-3:]),
            }
        )
    return result


def _assignment_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "cluster": _cluster_column(row),
            "pixels": row.get("Pixels", ""),
            "tissue_percent": row.get("% tissue", ""),
            "reliability": row.get("Reliability", ""),
            "current_domain": row.get("Domain", "Unassigned"),
            "enriched": row.get("Top markers (shrunk Z)", ""),
            "depleted": "not available in this older summary table",
        }
        for row in rows
    ]


def _spatial_text(row: Mapping[str, str]) -> str:
    return ", ".join(
        part
        for part in (
            f"location={row.get('Map location', '')}" if row.get("Map location") else "",
            f"centroid=({row.get('Centroid X (%)', '')}%, {row.get('Centroid Y (%)', '')}%)"
            if row.get("Centroid X (%)") or row.get("Centroid Y (%)")
            else "",
            f"span=({row.get('Horizontal span (%)', '')}% x {row.get('Vertical span (%)', '')}%)"
            if row.get("Horizontal span (%)") or row.get("Vertical span (%)")
            else "",
            f"map_edge={row.get('Touches map edge', '')}"
            if row.get("Touches map edge")
            else "",
            f"adjacent=[{row.get('Adjacent clusters', '')}]",
        )
        if part
    )


def prepare_cluster_labeling_evidence(
    prompt: str,
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve current-tab outputs and build bounded evidence for either model."""
    if not is_cluster_labeling_request(prompt):
        return None
    all_evidence = state.get(PAGE_EVIDENCE_KEY) or {}
    live = all_evidence.get("tab4") if isinstance(all_evidence, dict) else {}
    live = live if isinstance(live, dict) and live.get("section") == "pixel_classification" else {}

    explicit = _prompt_csv(prompt)
    live_assignment = str(live.get("assignment_csv") or "").strip()
    if explicit is None and not live_assignment:
        return {"error": "The Pixel Classification result table is not available yet. Run clustering once, then ask again."}
    assignment_raw = explicit or Path(live_assignment)
    try:
        assignment_path = require_assistant_read_path(assignment_raw, must_exist=True)
        assignment_rows = _read_csv(assignment_path)
    except (OSError, ValueError) as error:
        return {"error": f"I found the Pixel Classification table, but could not read it: {error}"}
    assignments = {_cluster_column(row): row for row in assignment_rows}

    table_paths = live.get("profile_tables") if isinstance(live.get("profile_tables"), dict) else {}
    candidate = Path(str(table_paths.get("shrunk_z") or "")) if table_paths else Path()
    if not candidate.is_file():
        sibling = assignment_path.with_name("step3_cluster_shrunk_z_values.csv")
        candidate = sibling if sibling.is_file() else Path()
    if candidate.is_file():
        try:
            profile_path = require_assistant_read_path(candidate, must_exist=True)
            clusters = _profile_summary(profile_path, assignments)
        except (OSError, ValueError):
            profile_path = None
            clusters = _assignment_summary(assignment_rows)
    else:
        profile_path = None
        clusters = _assignment_summary(assignment_rows)

    spatial_raw = str(table_paths.get("spatial") or "") if table_paths else ""
    spatial_candidate = Path(spatial_raw) if spatial_raw else assignment_path.with_name(
        "step3_cluster_spatial_summary.csv"
    )
    spatial_path: Path | None = None
    if spatial_candidate.is_file():
        try:
            spatial_path = require_assistant_read_path(spatial_candidate, must_exist=True)
            spatial_rows = {
                _cluster_column(row): row for row in _read_csv(spatial_path)
            }
            for cluster in clusters:
                cluster["spatial"] = _spatial_text(
                    spatial_rows.get(str(cluster.get("cluster") or ""), {})
                )
        except (OSError, ValueError):
            spatial_path = None
    for cluster in clusters:
        cluster.setdefault("spatial", "not available for this older run")

    figures = live.get("figures") if isinstance(live.get("figures"), dict) else {}
    if not figures:
        figures = {
            "balanced_heatmap": str(assignment_path.with_name("step3_cluster_balanced.png")),
            "cluster_map": str(assignment_path.with_name("step3_cluster_map.png")),
        }
    image_paths: list[Path] = []
    for key in ("balanced_heatmap", "cluster_map"):
        raw = str(figures.get(key) or "")
        if raw and Path(raw).is_file():
            try:
                image_paths.append(require_assistant_read_path(raw, must_exist=True))
            except (OSError, ValueError):
                pass

    rows_text = "\n".join(
        " | ".join(
            [
                f"cluster={item['cluster']}",
                f"pixels={item['pixels']}",
                f"tissue%={item['tissue_percent']}",
                f"reliability={item['reliability']}",
                f"current={item['current_domain']}",
                f"enriched: {item['enriched']}",
                f"depleted: {item['depleted']}",
                f"spatial: {item['spatial']}",
            ]
        )
        for item in clusters[:100]
    )
    note = (
        "\n\nCURRENT PIXEL CLASSIFICATION EVIDENCE (captured automatically from the open Annotation tab):\n"
        f"- Sample: {live.get('sample_id') or assignment_path.parent.name}\n"
        f"- Cluster count: {len(clusters)}\n"
        "- These are pixel/tissue-region clusters, not single-cell phenotypes. Suggest tissue-domain labels.\n"
        "- Use marker combinations, cluster reliability, tissue fraction, the balanced heatmap, and the spatial "
        "cluster map together. A single marker is not sufficient evidence.\n"
        "- If evidence images are attached, use the image legend supplied with them. They belong to this same "
        "clustering run. The numeric marker values below are the exact data behind the heatmap.\n"
        "- Return one Markdown table with: Cluster, proposed domain, confidence (high/medium/low), marker reason, "
        "and caution. Mention clusters that should be merged, excluded as artifact/low signal, or left unresolved.\n"
        "- Treat existing Domain values as editable user labels, not ground truth. Do not call a figure tool, do not "
        "ask for a CSV, and do not claim the labels were applied. Give the proposed labels now.\n\n"
        "CLUSTER EVIDENCE:\n" + rows_text
    )
    files = [
        assignment_path,
        *([profile_path] if profile_path is not None else []),
        *([spatial_path] if spatial_path is not None else []),
        *image_paths,
    ]
    return {
        "note": note,
        "images": image_paths,
        "files": files,
        "assignment_csv": assignment_path,
        "profile_csv": profile_path,
        "spatial_csv": spatial_path,
        "clusters": clusters,
    }
