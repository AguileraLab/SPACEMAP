"""Portable notebook records for SPACEMAP AI actions.

The notebook is a readable record, not a second job runner.  It keeps the exact
plan, status, results, progress messages, small previews, and table snapshots.
Large microscopy files stay on disk and are referenced by path.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable

from .action_monitor import action_details, action_log, progress_events
from .paths import is_assistant_read_allowed, require_assistant_write_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TABLE_SUFFIXES = {".csv", ".tsv"}
MAX_EMBEDDED_IMAGE_BYTES = 2 * 1024 * 1024
IMAGE_REENCODE_THRESHOLD = 512 * 1024
MAX_DISCOVERED_FILES = 500


def action_title(plan: dict[str, Any]) -> str:
    """Return a short human label that reflects the real saved action."""
    action_type = str(plan.get("action_type") or "")
    if action_type in {"registration", "register_codex_he_pair"}:
        return "CODEX + H&E registration"
    if action_type == "extract_rois":
        return "ROI extraction"
    if action_type == "optimize_blur_detection":
        return "Blur parameter test"
    if action_type == "optimize_artifact_detection":
        return "Bright glare/debris parameter test"
    return str(plan.get("title") or action_type.replace("_", " ").title() or "SPACEMAP action")


def _safe_existing_path(raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.startswith("/"):
        return None
    path = Path(raw).resolve(strict=False)
    try:
        return path if path.exists() and is_assistant_read_allowed(path) else None
    except OSError:
        return None


def _paths_in(value: Any, *, key: str = "") -> Iterable[tuple[str, Path]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _paths_in(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _paths_in(child, key=key)
    else:
        path = _safe_existing_path(value)
        if path is not None:
            yield key, path


def collect_action_outputs(details: dict[str, Any]) -> dict[str, list[Path]]:
    """Find the selected action's outputs without scanning unrelated projects."""
    plan = details.get("plan") or {}
    report = details.get("report") or {}
    found: dict[str, Path] = {}

    # Report paths are produced outputs.  Plan paths are included only when the
    # field name clearly describes an output, preview, temporary result, or QC.
    for _key, path in _paths_in(report):
        found[str(path)] = path
    output_words = ("output", "final", "preview", "qc", "staging", "intermediate", "backup")
    for key, path in _paths_in(plan):
        if any(word in key.casefold() for word in output_words):
            found[str(path)] = path

    # Parameter experiments create useful files gradually.  Scan only their
    # unique per-action run folder so live previews appear while the job runs.
    run_root = _safe_existing_path(plan.get("run_output_dir"))
    if run_root is not None and run_root.is_dir():
        count = 0
        try:
            for path in run_root.rglob("*"):
                if count >= MAX_DISCOVERED_FILES:
                    break
                if path.is_file() and is_assistant_read_allowed(path):
                    found[str(path.resolve())] = path.resolve()
                    count += 1
        except OSError:
            pass

    groups: dict[str, list[Path]] = {"images": [], "tables": [], "data": [], "large": [], "other": []}
    for path in found.values():
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        name = path.name.casefold()
        if suffix in IMAGE_SUFFIXES:
            group = "images"
        elif suffix in TABLE_SUFFIXES:
            group = "tables"
        elif suffix in {".json", ".geojson"}:
            group = "data"
        elif name.endswith((".ome.tiff", ".ome.tif", ".qptiff", ".tiff", ".tif")):
            group = "large"
        else:
            group = "other"
        groups[group].append(path)
    for paths in groups.values():
        paths.sort(key=lambda item: (item.stat().st_mtime_ns if item.exists() else 0, item.name), reverse=True)
    return groups


def read_table_preview(path: Path, *, maximum_rows: int = 100) -> list[dict[str, str]]:
    """Read a small, dependency-free CSV/TSV preview."""
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            return [dict(row) for _, row in zip(range(maximum_rows), csv.DictReader(stream, delimiter=delimiter))]
    except (OSError, csv.Error):
        return []


def _markdown_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(source: str, outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True),
    }


def _plain_output(text: str) -> dict[str, Any]:
    return {"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}


def _image_output(path: Path) -> dict[str, Any] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    mime = "image/jpeg" if path.suffix.casefold() in {".jpg", ".jpeg"} else "image/png"
    if len(data) > IMAGE_REENCODE_THRESHOLD:
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as opened:
                image = opened.convert("RGB")
                image.thumbnail((1_400, 1_400), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=82, optimize=True)
                data = buffer.getvalue()
                mime = "image/jpeg"
        except (OSError, ValueError):
            return None
    if len(data) > MAX_EMBEDDED_IMAGE_BYTES:
        return None
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {mime: base64.b64encode(data).decode("ascii"), "text/plain": [path.name]},
    }


def _table_text(path: Path) -> str:
    rows = read_table_preview(path, maximum_rows=50)
    if not rows:
        return f"{path.name}: no readable table rows."
    columns = list(rows[0])
    widths = {name: max(len(name), *(len(str(row.get(name, ""))) for row in rows)) for name in columns}
    widths = {name: min(value, 40) for name, value in widths.items()}

    def line(row: dict[str, Any]) -> str:
        return " | ".join(str(row.get(name, ""))[: widths[name]].ljust(widths[name]) for name in columns)

    return "\n".join([line({name: name for name in columns}), line({name: "-" * widths[name] for name in columns}), *(line(row) for row in rows)])


def _replay_source(plan: dict[str, Any]) -> str:
    """Return a guarded, standalone replay cell for the saved action."""
    action_type = str(plan.get("action_type") or "")
    header = (
        "# Set this to True only when you want to repeat the action.\n"
        "# A new output path is used so the completed run is not overwritten.\n"
        "RUN_ACTION = False\n\n"
    )
    if action_type in {"optimize_blur_detection", "optimize_artifact_detection"}:
        runner = "run_bright_parameter_search" if action_type == "optimize_artifact_detection" else "run_blur_parameter_search"
        return header + (
            "from pathlib import Path\n"
            f"from spacemap_workflows.annotation import {runner}\n\n"
            "REPLAY_OUTPUT_DIR = Path(PLAN['run_output_dir'] + '_notebook_rerun')\n"
            "if RUN_ACTION:\n"
            f"    replay_result = {runner}(\n"
            "        PLAN['image_path'], PLAN['tissue_boundary_path'], REPLAY_OUTPUT_DIR,\n"
            "        project_config_path=PLAN.get('project_config_path') or None,\n"
            "        microns_per_pixel=PLAN.get('microns_per_pixel'),\n"
            "        maximum_trials=PLAN.get('maximum_trials', 10),\n"
            "    )\n"
            "    replay_result\n"
        )
    if action_type == "extract_rois":
        return header + (
            "from pathlib import Path\n"
            "from spacemap.registration import process_single_image_rois\n\n"
            "REPLAY_OUTPUT_DIR = Path(PLAN['output_dir'] + '_notebook_rerun')\n"
            "if RUN_ACTION:\n"
            "    REPLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)\n"
            "    process_single_image_rois(\n"
            "        PLAN['image_path'], PLAN['boundaries_path'], str(REPLAY_OUTPUT_DIR),\n"
            "        {'codex': PLAN['mpp']},\n"
            "        channel_config={'keep': PLAN['channels'], 'rename': {}, 'colors': {}},\n"
            "    )\n"
        )
    if action_type in {"registration", "register_codex_he_pair"}:
        return header + (
            "from pathlib import Path\n"
            "from spacemap.registration import (extract_dapi_channel, merge_multi_round_ome_tiff, "
            "preprocess_he_image, register_images)\n\n"
            "REPLAY_OUTPUT_DIR = Path(PLAN['output_dir'] + '_notebook_rerun')\n"
            "if RUN_ACTION:\n"
            "    REPLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)\n"
            "    work = REPLAY_OUTPUT_DIR / 'intermediate'\n"
            "    work.mkdir()\n"
            "    he = preprocess_he_image(PLAN['he_path'], str(work), "
            "{'hne': PLAN['he_mpp'], 'codex': PLAN['codex_mpp']}, overwrite=False, "
            "pre_rotation_deg=PLAN['pre_rotation_deg'])\n"
            "    ref = extract_dapi_channel(PLAN['codex_path'], str(work), overwrite=False, "
            "reference_channel=PLAN['reference_channel'], reference_channel_index=PLAN['reference_channel_index'])\n"
            "    registered = register_images(he, ref, str(work), "
            "{'hne': PLAN['codex_mpp'], 'codex': PLAN['codex_mpp']}, overwrite=False, "
            "reference_channel_index=0, n_keypoints=PLAN['n_keypoints'])\n"
            "    final = REPLAY_OUTPUT_DIR / (PLAN['prefix'] + '_merged.ome.tiff')\n"
            "    merge_multi_round_ome_tiff([PLAN['codex_path'], registered], str(final), "
            "PLAN['codex_mpp'], channel_configs=None, overwrite=False)\n"
            "    final\n"
        )
    if action_type == "general_spacemap_job" and plan.get("code"):
        return header + (
            "from pathlib import Path\n"
            "INPUTS = dict(PLAN['inputs'])\n"
            "PARAMETERS = dict(PLAN.get('parameters') or {})\n"
            "OUTPUT_DIR = Path(PLAN['output_dir'] + '_notebook_rerun')\n"
            "if RUN_ACTION:\n"
            "    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)\n"
            + "\n".join("    " + line for line in str(plan["code"]).splitlines())
            + "\n"
        )
    return header + "# This older action type has no automatic replay recipe.\n"


def build_action_notebook(action_id: str) -> dict[str, Any]:
    """Build one self-contained, viewable notebook record for an action."""
    details = action_details(action_id)
    if not details.get("plan"):
        raise ValueError("The selected action could not be found.")
    plan = details["plan"]
    status = details.get("status") or {}
    report = details.get("report") or {}
    events = progress_events(action_id, limit=2_000)
    outputs = collect_action_outputs(details)
    title = action_title(plan)
    request = str(plan.get("user_request") or "No original chat request was saved.")
    state = str(status.get("state") or "unknown")
    stage = str(status.get("stage") or "")
    cells: list[dict[str, Any]] = [
        _markdown_cell(
            f"# {title}\n\n"
            f"**Action:** `{action_id}`  \n**State:** {state}  \n**Stage:** {stage}  \n"
            f"**Created:** {plan.get('created_at', '')}\n\n"
            "This notebook is the durable record of one SPACEMAP AI action. Large source and output "
            "images are referenced by path and are not copied into the notebook."
        ),
        _markdown_cell(f"## User request\n\n{request}"),
        _code_cell(
            "# Exact saved inputs and parameters\n"
            "import json\n"
            f"PLAN = json.loads({json.dumps(plan, indent=2, default=str)!r})\n"
            f"STATUS = json.loads({json.dumps(status, indent=2, default=str)!r})\n"
            f"REPORT = json.loads({json.dumps(report, indent=2, default=str)!r})\n"
            "PLAN, STATUS, REPORT",
            [_plain_output("The exact plan, final status, and report are stored in PLAN, STATUS, and REPORT.\n")],
        ),
    ]

    comparison = ((report.get("result") or {}).get("comparison") or report.get("comparison") or [])
    if comparison:
        cells.append(_markdown_cell("## Tested settings and measurements"))
        cells.append(
            _code_cell(
                "# The original comparison table is available as Python data.\n"
                f"COMPARISON = {json.dumps(comparison, indent=2, default=str)}\n"
                "COMPARISON",
                [_plain_output(json.dumps(comparison[:20], indent=2, default=str) + "\n")],
            )
        )

    if outputs["tables"]:
        cells.append(_markdown_cell("## Output tables"))
        for path in outputs["tables"][:10]:
            cells.append(_markdown_cell(f"### {path.name}\n\n`{path}`"))
            cells.append(_code_cell(f"# Preview saved from {path}\n", [_plain_output(_table_text(path) + "\n")]))

    notebook_images = outputs["images"]
    plain_images = [path for path in notebook_images if "scale_bar" not in path.stem.casefold()]
    if plain_images and not bool(plan.get("include_scale_bars")):
        notebook_images = plain_images
    embedded = 0
    for path in notebook_images:
        output = _image_output(path)
        if output is None:
            continue
        if embedded == 0:
            cells.append(_markdown_cell("## Figures and QC previews"))
        cells.append(_markdown_cell(f"### {path.name}\n\n`{path}`"))
        cells.append(_code_cell(f"# Saved preview: {path}\n", [output]))
        embedded += 1
        if embedded >= 12:
            break

    referenced = [*outputs["large"], *outputs["data"], *outputs["other"]]
    if referenced:
        cells.append(
            _markdown_cell(
                "## Output files\n\n" + "\n".join(f"- `{path}`" for path in referenced[:100])
            )
        )

    event_text = "\n".join(
        f"{item.get('created_at') or item.get('time') or ''}  {item.get('stage', '')}: {item.get('message', '')}"
        for item in events
    )
    log_text = action_log(action_id, max_characters=200_000)
    cells.extend(
        [
            _markdown_cell(
                "## Repeat this action\n\n"
                "This cell uses the same SPACEMAP functions and settings but writes to a new replay folder. "
                "It does nothing until `RUN_ACTION` is changed to `True`."
            ),
            _code_cell(_replay_source(plan)),
            _markdown_cell("## Progress and technical record"),
            _code_cell("# Recorded progress messages\n", [_plain_output((event_text or "No progress messages.") + "\n")]),
            _code_cell("# Worker output\n", [_plain_output((log_text or "No worker output.") + "\n")]),
            _markdown_cell(
                "## Reuse this run\n\n"
                "Open the notebook in the SPACEMAP environment. The `PLAN`, `STATUS`, `REPORT`, and "
                "`COMPARISON` data (when present) can be analyzed directly. The guarded replay cell "
                "can repeat the scientific work into a new folder without replacing this run."
            ),
        ]
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "SPACEMAP AI", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "spacemap_ai": {"schema_version": 1, "action_id": action_id, "action_type": plan.get("action_type")},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def action_notebook_json(action_id: str) -> str:
    return json.dumps(build_action_notebook(action_id), indent=1, ensure_ascii=False)


def save_action_notebook(action_id: str) -> Path:
    details = action_details(action_id)
    directory = require_assistant_write_path(details.get("directory") or "")
    path = require_assistant_write_path(directory / f"SPACEMAP_action_{action_id[:8]}.ipynb")
    path.write_text(action_notebook_json(action_id), encoding="utf-8")
    return path


def notebook_action_id(payload: bytes | str) -> str:
    raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    document = json.loads(raw)
    return str(((document.get("metadata") or {}).get("spacemap_ai") or {}).get("action_id") or "")
