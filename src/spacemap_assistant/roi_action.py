"""Reviewed ROI extraction action backed by SPACEMAP's existing ROI code."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .paths import (
    AI_ROOT,
    SOURCE_ROOT,
    ASSISTANT_DATA_DIR,
    action_workspace_root,
    is_within,
    require_assistant_read_path,
    require_assistant_write_path,
)
from .registration_action import SUPPORTED_TIFF_SUFFIXES, _file_signature, utc_now


ACTIONS_DIR = ASSISTANT_DATA_DIR / "actions"


class ROIActionError(RuntimeError):
    """A safe, user-readable ROI action failure."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    target = require_assistant_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_action_id(action_id: str) -> str:
    try:
        parsed = uuid.UUID(str(action_id))
    except (ValueError, TypeError, AttributeError) as error:
        raise ROIActionError("The ROI action number is invalid.") from error
    if str(parsed) != str(action_id):
        raise ROIActionError("The ROI action number is invalid.")
    return str(parsed)


def action_directory(action_id: str) -> Path:
    return require_assistant_write_path(ACTIONS_DIR / _validated_action_id(action_id))


def _plan_digest(action_id: str) -> str:
    try:
        return hashlib.sha256((action_directory(action_id) / "plan.json").read_bytes()).hexdigest()
    except OSError as error:
        raise ROIActionError("The saved ROI extraction plan is missing.") from error


def _is_supported_tiff(path: Path) -> bool:
    return any(path.name.lower().endswith(suffix) for suffix in SUPPORTED_TIFF_SUFFIXES)


def _validate_inputs(
    image_path: str | Path,
    boundaries_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    image = require_assistant_read_path(image_path, must_exist=True)
    boundaries = require_assistant_read_path(boundaries_path, must_exist=True)
    output = require_assistant_write_path(output_dir)
    if not image.is_file() or not _is_supported_tiff(image):
        raise ROIActionError("The registered image must be an existing TIFF or QPTIFF file.")
    if not boundaries.is_file() or boundaries.suffix.lower() not in {".geojson", ".json"}:
        raise ROIActionError("The ROI boundaries must be an existing GeoJSON file.")

    output_project = action_workspace_root(output)
    relative_output = output.relative_to(output_project)
    if not relative_output.parts or relative_output.parts[0].lower() != "outputs":
        raise ROIActionError("The ROI output folder must be inside the project's outputs folder.")
    if is_within(output, image.parent) or is_within(output, boundaries.parent):
        raise ROIActionError("The output folder cannot be inside an input folder.")
    return image, boundaries, output, output_project


def _iter_points(value: Any) -> Iterable[tuple[float, float]]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            x, y = float(value[0]), float(value[1])
            if math.isfinite(x) and math.isfinite(y):
                yield x, y
            return
        for item in value:
            yield from _iter_points(item)


def _roi_label(feature: dict[str, Any], index: int) -> str:
    properties = feature.get("properties") or {}
    for key in ("label", "name"):
        value = properties.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    classification = properties.get("classification")
    if isinstance(classification, dict):
        value = classification.get("name")
        if value is not None and str(value).strip():
            return str(value).strip()
    value = properties.get("id")
    if value is not None and str(value).strip():
        return str(value).strip()
    return f"ROI_{index + 1}"


def _safe_label(label: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in label)


def _image_stem(path: Path) -> str:
    stem = path.stem
    if stem.lower().endswith(".ome"):
        stem = stem[:-4]
    if stem.lower().endswith("_merged"):
        stem = stem[:-7]
    return stem or "image"


def _inspect_geojson(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ROIActionError("The ROI GeoJSON could not be read.") from error
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise ROIActionError("The ROI file must be a GeoJSON FeatureCollection.")

    records: list[dict[str, Any]] = []
    used_labels: set[str] = set()
    for index, feature in enumerate(payload["features"]):
        geometry = (feature or {}).get("geometry") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        points = list(_iter_points(geometry.get("coordinates")))
        if len(points) < 3:
            raise ROIActionError(f"ROI {index + 1} does not contain a valid polygon.")
        xs, ys = zip(*points)
        raw_box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        box = [
            max(0, min(raw_box[0], width - 1)),
            max(0, min(raw_box[1], height - 1)),
            max(1, min(raw_box[2], width)),
            max(1, min(raw_box[3], height)),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ROIActionError(f"ROI {index + 1} is outside the registered image.")
        label = _roi_label(feature, index)
        safe = _safe_label(label)
        if not safe:
            raise ROIActionError(f"ROI {index + 1} has no usable label.")
        if safe.lower() in used_labels:
            raise ROIActionError(f"Two ROI labels create the same output name: {safe}")
        used_labels.add(safe.lower())
        records.append({"label": label, "safe_label": safe, "bbox": box})
    if not records:
        raise ROIActionError("The GeoJSON contains no Polygon or MultiPolygon ROI boundaries.")
    if len(records) > 2_000:
        raise ROIActionError("The GeoJSON contains more than 2,000 ROIs. Split it into smaller jobs.")
    return records


def inspect_image(path: Path) -> dict[str, Any]:
    """Read image dimensions, channels, and MPP without loading the full image."""
    import numpy as np
    import tifffile
    from spacemap.background_subtraction_utils import get_channel_info, get_image_mpp

    with tifffile.TiffFile(path) as tif:
        if not tif.series and not tif.pages:
            raise ROIActionError("The registered TIFF has no readable image data.")
        if tif.series:
            series = max(
                tif.series,
                key=lambda item: int(np.prod(getattr(item, "shape", (0,)))),
            )
            axes = str(getattr(series, "axes", "") or "")
            shape = tuple(int(value) for value in series.shape)
            if "Y" in axes and "X" in axes:
                height, width = shape[axes.index("Y")], shape[axes.index("X")]
            else:
                height, width = tif.pages[0].shape[:2]
        else:
            height, width = tif.pages[0].shape[:2]
    names, _colors, _metadata, metadata_mpp = get_channel_info(str(path))
    detected_mpp = get_image_mpp(str(path)).get("codex")
    mpp = detected_mpp if detected_mpp is not None else metadata_mpp.get("codex")
    return {
        "width": int(width),
        "height": int(height),
        "channels": [str(name) for name in names],
        "mpp": None if mpp is None else float(mpp),
    }


@dataclass(frozen=True)
class ROIPlan:
    schema_version: int
    action_id: str
    action_type: str
    created_at: str
    project_root: str
    image_path: str
    boundaries_path: str
    output_dir: str
    image_width: int
    image_height: int
    mpp: float
    channels: list[str]
    rois: list[dict[str, Any]]
    overwrite: bool
    image_signature: dict[str, int]
    boundaries_signature: dict[str, int]
    staging_dir: str
    qc_preview: str
    conversation_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_roi_plan(
    image_path: str | Path,
    boundaries_path: str | Path,
    output_dir: str | Path,
    *,
    mpp: float | None = None,
    overwrite: bool = False,
    conversation_id: str = "",
) -> ROIPlan:
    image, boundaries, output, project_root = _validate_inputs(
        image_path, boundaries_path, output_dir
    )
    image_info = inspect_image(image)
    resolved_mpp = float(mpp if mpp is not None and float(mpp) > 0 else image_info["mpp"] or 0)
    if resolved_mpp <= 0:
        raise ROIActionError("MPP could not be read. Enter a valid microns-per-pixel value.")
    if not image_info["channels"]:
        raise ROIActionError("No image channels could be read from the registered TIFF.")

    roi_records = _inspect_geojson(boundaries, image_info["width"], image_info["height"])
    action_id = str(uuid.uuid4())
    stem = _image_stem(image)
    for record in roi_records:
        final = output / f"{stem}_{record['safe_label']}.ome.tiff"
        record["output_path"] = str(final)
        record["existing_output_signature"] = _file_signature(final) if final.exists() else None
        record["backup_path"] = str(
            output / f"{stem}_{record['safe_label']}.before-{action_id[:8]}.ome.tiff"
        )
    existing = [record["output_path"] for record in roi_records if record["existing_output_signature"]]
    if existing and not overwrite:
        joined = "\n".join(existing[:5])
        extra = f"\n...and {len(existing) - 5} more." if len(existing) > 5 else ""
        raise ROIActionError(
            "ROI output files already exist. Ask to overwrite them or select "
            f"'replace existing outputs'.\n{joined}{extra}"
        )

    run_dir = output / "_spacemap_ai_actions" / action_id
    plan = ROIPlan(
        schema_version=1,
        action_id=action_id,
        action_type="extract_rois",
        created_at=utc_now(),
        project_root=str(project_root.resolve()),
        image_path=str(image),
        boundaries_path=str(boundaries),
        output_dir=str(output),
        image_width=image_info["width"],
        image_height=image_info["height"],
        mpp=resolved_mpp,
        channels=image_info["channels"],
        rois=roi_records,
        overwrite=bool(overwrite),
        image_signature=_file_signature(image),
        boundaries_signature=_file_signature(boundaries),
        staging_dir=str(run_dir / "staging"),
        qc_preview=str(run_dir / "roi_extraction_qc.png"),
        conversation_id=str(conversation_id).strip(),
    )
    directory = action_directory(action_id)
    _atomic_json(directory / "plan.json", plan.as_dict())
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": "planned",
            "stage": "Waiting for approval",
            "created_at": plan.created_at,
            "updated_at": utc_now(),
        },
    )
    return plan


def load_roi_plan(action_id: str, *, verify_inputs: bool = False) -> ROIPlan:
    directory = action_directory(action_id)
    try:
        plan = ROIPlan(**json.loads((directory / "plan.json").read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ROIActionError("The saved ROI extraction plan is missing or invalid.") from error
    if plan.action_id != action_id or plan.action_type != "extract_rois":
        raise ROIActionError("The saved ROI extraction plan failed its safety check.")
    image, boundaries, output, project_root = _validate_inputs(
        plan.image_path, plan.boundaries_path, plan.output_dir
    )
    if str(project_root.resolve()) != plan.project_root:
        raise ROIActionError("The saved project boundary no longer matches the plan.")
    for generated in (
        plan.staging_dir,
        plan.qc_preview,
        *(record["output_path"] for record in plan.rois),
        *(record["backup_path"] for record in plan.rois),
    ):
        if not is_within(generated, output):
            raise ROIActionError("A generated ROI path escaped the approved output folder.")
    if verify_inputs and (
        _file_signature(image) != plan.image_signature
        or _file_signature(boundaries) != plan.boundaries_signature
    ):
        raise ROIActionError("An input changed after the plan was created. Build a new plan.")
    return plan


def load_action_status(action_id: str) -> dict[str, Any]:
    try:
        return json.loads((action_directory(action_id) / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"action_id": action_id, "state": "unknown", "stage": "Status is unavailable"}


def list_roi_actions(limit: int = 25) -> list[dict[str, Any]]:
    if not ACTIONS_DIR.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for plan_path in ACTIONS_DIR.glob("*/plan.json"):
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            if payload.get("action_type") != "extract_rois":
                continue
            action_id = _validated_action_id(plan_path.parent.name)
            plan = load_roi_plan(action_id)
            status = load_action_status(action_id)
        except (ROIActionError, OSError, ValueError, json.JSONDecodeError):
            continue
        items.append(
            {
                "action_id": action_id,
                "created_at": plan.created_at,
                "image_name": Path(plan.image_path).name,
                "roi_count": len(plan.rois),
                "state": status.get("state", "unknown"),
            }
        )
    return sorted(items, key=lambda item: item["created_at"], reverse=True)[: max(1, int(limit))]


def launch_roi_action(action_id: str, *, approval_method: str) -> int:
    """Record approval and start only the fixed ROI worker."""
    plan = load_roi_plan(action_id, verify_inputs=True)
    directory = action_directory(action_id)
    if load_action_status(action_id).get("state") != "planned":
        raise ROIActionError("This ROI plan is not waiting for approval and cannot start again.")
    for record in plan.rois:
        final = Path(record["output_path"])
        expected = record["existing_output_signature"]
        if final.exists() and not plan.overwrite:
            raise ROIActionError("An ROI output now exists, so this action was stopped.")
        if plan.overwrite:
            if expected is None and final.exists():
                raise ROIActionError("An unreviewed ROI output appeared after planning.")
            if expected is not None and (
                not final.exists() or _file_signature(final) != expected
            ):
                raise ROIActionError("An existing ROI output changed after review.")

    approved_at = utc_now()
    _atomic_json(
        directory / "approval.json",
        {
            "action_id": action_id,
            "approved_at": approved_at,
            "approval_method": str(approval_method),
            "plan_sha256": _plan_digest(action_id),
            "replace_existing_outputs": plan.overwrite,
        },
    )
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": "launching",
            "stage": "Starting ROI worker",
            "created_at": plan.created_at,
            "approved_at": approved_at,
            "updated_at": utc_now(),
        },
    )
    environment = os.environ.copy()
    source_root = SOURCE_ROOT
    previous_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(source_root) + (
        os.pathsep + previous_pythonpath if previous_pythonpath else ""
    )
    environment["MPLBACKEND"] = "Agg"
    command = [sys.executable, "-m", "spacemap_assistant.roi_worker", "--action-id", action_id]
    log_path = directory / "worker.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(AI_ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _atomic_json(
        directory / "launcher.json",
        {
            "action_id": action_id,
            "pid": int(process.pid),
            "launched_at": utc_now(),
            "worker": "spacemap_assistant.roi_worker",
            "log_path": str(log_path),
        },
    )
    return int(process.pid)


def recent_action_log(action_id: str, max_characters: int = 30_000) -> str:
    try:
        content = (action_directory(action_id) / "worker.log").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return "The worker has not written a log yet."
    return content[-max(1_000, int(max_characters)) :]


def parse_roi_extraction_request(prompt: str) -> dict[str, Any] | None:
    """Extract clear image, GeoJSON, and output paths from an ROI request."""
    cleaned = str(prompt).replace("\\_", "_")
    lowered = cleaned.lower()
    if "roi" not in lowered or not any(term in lowered for term in ("extract", "crop", "create")):
        return None
    raw_paths = re.findall(r"(?<![A-Za-z0-9])(/[^\s`\"']+)", cleaned)
    paths = [value.rstrip(".,;:?!)]}") for value in raw_paths]
    image = next((value for value in paths if _is_supported_tiff(Path(value))), "")
    boundaries = next(
        (value for value in paths if Path(value).suffix.lower() in {".geojson", ".json"}),
        "",
    )
    output = next(
        (
            value
            for value in paths
            if value not in {image, boundaries} and "/outputs/" in value.lower()
        ),
        "",
    )
    if not image or not boundaries or not output:
        return None
    overwrite = any(term in lowered for term in ("overwrite", "over-write", "replace existing"))
    return {
        "image_path": image,
        "boundaries_path": boundaries,
        "output_dir": output,
        "overwrite": overwrite,
    }
