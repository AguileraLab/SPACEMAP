"""UI-independent annotation workflows used by the SPACEMAP agent."""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SENSITIVITY_SCALES = {1: 0.50, 2: 0.63, 3: 0.79, 4: 1.00, 5: 1.26, 6: 1.59, 7: 2.00}


def default_bright_trials(base_artifact_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Ten bounded glare/debris tests around the two scientific thresholds."""
    trials: list[dict[str, Any]] = [{"name": "current_project_settings", "overrides": {}}]
    # Texture 2.0 is the normal project default, so it is already represented
    # by the first control trial. These nine combinations are all distinct on
    # a default project and explore stricter and more permissive smoothness.
    for brightness in (99.0, 99.5, 99.8):
        for texture in (1.0, 3.0, 5.0):
            trials.append(
                {
                    "name": f"bright_p{str(brightness).replace('.', '_')}_texture_{str(texture).replace('.', '_')}",
                    "overrides": {
                        "detect_bright_artifacts": True,
                        "use_breadth_signal": False,
                        "bright_flat_mean_percentile": brightness,
                        "bright_flat_texture_percentile": texture,
                    },
                }
            )
    return trials


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_id(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".ome.tiff", ".ome.tif", ".qptiff", ".tiff", ".tif"):
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _project_tab_settings(project_config_path: str | Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    if not project_config_path:
        return {}, {}, {}
    try:
        payload = json.loads(Path(project_config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}, {}, {}
    tab = ((payload.get("tab_configurations") or {}).get("tab4_annotation") or {})
    tissue = tab.get("annot_ta_tissue_config") or {}
    artifacts = tab.get("annot_ta_artifact_config") or {}
    markers = tab.get("annot_ta_artifact_markers") or {}
    return (
        dict(tissue) if isinstance(tissue, dict) else {},
        dict(artifacts) if isinstance(artifacts, dict) else {},
        {str(key): bool(value) for key, value in markers.items()} if isinstance(markers, dict) else {},
    )


def default_blur_trials(base_artifact_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """A bounded search that also tests the upstream population-separation gate."""
    trials: list[dict[str, Any]] = [
        {"name": "current_project_settings", "overrides": {}},
    ]
    for level, rule in (
        (2, "both"),
        (4, "both"),
        (6, "both"),
        (2, "either"),
        (4, "either"),
        (6, "either"),
    ):
        trials.append(
            {
                "name": f"auto_level_{level}_{rule}",
                "overrides": {
                    "blur_threshold_mode": "auto",
                    "blur_reference_mode": "he_and_self",
                    "blur_sensitivity_level": level,
                    "blur_sensitivity": SENSITIVITY_SCALES[level],
                    "blur_evidence_rule": rule,
                    "require_codex_signal": True,
                    "use_marker_consensus": True,
                },
            }
        )
    # Sensitivity is applied only after SPACEMAP finds a separated blur-score
    # population. A strict project/default ``blur_min_separation`` can therefore
    # make every sensitivity test identical and empty. Keep the search bounded,
    # but spend the final three trials testing that important upstream gate.
    for gap, rule in ((2.5, "either"), (2.0, "either"), (2.5, "pooled_only")):
        trials.append(
            {
                "name": f"auto_relaxed_gap_{str(gap).replace('.', '_')}_{rule}",
                "overrides": {
                    "blur_threshold_mode": "auto",
                    "blur_reference_mode": "he_and_self",
                    "blur_sensitivity_level": 4,
                    "blur_sensitivity": SENSITIVITY_SCALES[4],
                    "blur_evidence_rule": rule,
                    "blur_min_separation": gap,
                    "require_codex_signal": True,
                    "use_marker_consensus": True,
                },
            }
        )
    # A pre-existing project can carry values added by an older release. They
    # are kept in the baseline and overridden only where the trial says so.
    return trials


def _feature_union(features: list[dict[str, Any]]):
    if not features:
        return None
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union

        geometries = [shape(item["geometry"]) for item in features if item.get("geometry")]
        return unary_union(geometries) if geometries else None
    except Exception:
        return None


def _stability_scores(feature_sets: list[list[dict[str, Any]]]) -> list[float]:
    unions = [_feature_union(features) for features in feature_sets]
    scores: list[float] = []
    for index, geometry in enumerate(unions):
        if geometry is None or geometry.is_empty:
            scores.append(0.0)
            continue
        overlaps: list[float] = []
        for other_index, other in enumerate(unions):
            if other_index == index or other is None or other.is_empty:
                continue
            denominator = float(geometry.union(other).area)
            if denominator > 0:
                overlaps.append(float(geometry.intersection(other).area) / denominator)
        scores.append(max(overlaps, default=0.0))
    return scores


def _machine_screen_score(row: dict[str, Any]) -> float:
    """Rank candidates for review; this is not a biological ground truth score."""
    if int(row.get("polygon_count") or 0) == 0:
        return 0.20 if not row.get("blur_population_found") else 0.05
    confidence = float(row.get("mean_confidence") or 0.0)
    severity = float(row.get("mean_severity") or 0.0)
    stability = float(row.get("stability_iou") or 0.0)
    gap = min(float(row.get("separation_gap") or 0.0) / 8.0, 1.0)
    blurry_fraction = float(row.get("blurry_percent") or 0.0) / 100.0
    preservation = max(0.0, 1.0 - min(blurry_fraction / 0.30, 1.0))
    return round(0.30 * confidence + 0.20 * severity + 0.25 * stability + 0.15 * gap + 0.10 * preservation, 6)


def _bright_machine_screen_score(row: dict[str, Any]) -> float:
    """Prioritize stable, tissue-preserving candidates for visual review."""
    if int(row.get("polygon_count") or 0) == 0:
        return 0.0
    stability = float(row.get("stability_iou") or 0.0)
    bright_fraction = max(0.0, float(row.get("bright_percent") or 0.0))
    preservation = max(0.0, 1.0 - min(bright_fraction / 5.0, 1.0))
    # Marker signal is intentionally not treated as proof of an artifact. A
    # structured bright marker can instead reveal a biological false positive;
    # that evidence is carried into the visual-review tiles.
    return round(0.70 * stability + 0.30 * preservation, 6)


def run_blur_parameter_search(
    image_path: str | Path,
    tissue_boundary_path: str | Path,
    output_dir: str | Path,
    *,
    project_config_path: str | Path | None = None,
    microns_per_pixel: float | None = None,
    trials: list[dict[str, Any]] | None = None,
    maximum_trials: int = 10,
    progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run bounded SPACEMAP blur trials and save a provisional review package.

    The function deliberately stops before accepting/rejecting polygons. A
    deterministic score selects which candidate the visual model should inspect
    first; the selected AI or a person then performs the independent review.
    """
    from spacemap.tissue_annotation_utils import (
        DEFAULT_ARTIFACT_CONFIG,
        DEFAULT_TISSUE_CONFIG,
        PyramidImageReader,
        detect_artifacts,
        detect_tissue_mask,
        load_geojson_mask,
        region_features_from_labels,
        split_he_and_codex_channels,
        write_feature_collection,
    )

    image = Path(image_path).resolve()
    boundary = Path(tissue_boundary_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    trial_root = destination / "trials"
    trial_root.mkdir()
    sample = _sample_id(image)
    emit = progress or (lambda _stage, _message: None)

    saved_tissue, saved_artifacts, selected_markers = _project_tab_settings(project_config_path)
    tissue_config = {**DEFAULT_TISSUE_CONFIG, **saved_tissue}
    if microns_per_pixel is not None and float(microns_per_pixel) > 0:
        tissue_config["microns_per_pixel"] = float(microns_per_pixel)
    tissue_config["detect_blood_clot"] = False
    base_artifact = {**DEFAULT_ARTIFACT_CONFIG, **saved_artifacts}
    # This workflow is only optimizing blur. Avoid spending time recomputing
    # bright/lint outputs during every parameter trial.
    base_artifact["detect_bright_artifacts"] = False
    base_artifact["detect_filamentous_lint"] = False
    base_artifact["write_review_manifest"] = False

    reader = PyramidImageReader(image)
    he_indices, codex_indices = split_he_and_codex_channels(reader.channel_names)
    if selected_markers:
        codex_indices = [
            index
            for index in codex_indices
            if selected_markers.get(str(reader.channel_names[index]), False)
        ]
    if not codex_indices:
        raise ValueError("No selected CODEX marker channels were found for blur detection.")
    marker_names = [str(reader.channel_names[index]) for index in codex_indices]

    emit("Reading image", f"Reading {len(codex_indices)} selected CODEX channels once.")
    last_read = -1

    def image_progress(done: int, total: int, name: str) -> None:
        nonlocal last_read
        if done == last_read:
            return
        last_read = done
        emit("Reading image", f"Channel {done}/{total}: {name}")

    maps = detect_tissue_mask(
        reader,
        tissue_config,
        he_indices=he_indices,
        codex_indices=codex_indices,
        progress=image_progress,
        need_codex_maps=True,
    )
    factor = int(tissue_config.get("blur_downsample", 2))
    tissue_mask = load_geojson_mask(boundary, tuple(maps["shape"]), float(factor))
    if not tissue_mask.any():
        raise ValueError("The tissue-boundary GeoJSON produced an empty mask.")

    requested_trials = list(trials or default_blur_trials(base_artifact))
    limit = min(20, max(1, int(maximum_trials)))
    requested_trials = requested_trials[:limit]
    feature_sets: list[list[dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    trial_records: list[dict[str, Any]] = []
    for position, trial in enumerate(requested_trials, start=1):
        name = str(trial.get("name") or f"trial_{position:02d}")
        overrides = dict(trial.get("overrides") or {})
        config = {**base_artifact, **overrides}
        emit("Testing blur settings", f"Trial {position}/{len(requested_trials)}: {name}")
        artifacts = detect_artifacts(
            tissue_mask=tissue_mask,
            codex_projection=maps["codex_projection"],
            he_gray=maps["he_gray"],
            codex_robust_max=maps["codex_robust_max"],
            channel_positive_count=maps["channel_positive_count"],
            n_codex_channels=len(codex_indices),
            config=config,
            microns_per_pixel=float(tissue_config["microns_per_pixel"]),
            downsample=factor,
            border_px=int(maps.get("border_px", 0)),
            marker_names=marker_names,
            channel_focus_grid=maps.get("channel_focus_grid"),
            channel_signal_grid=maps.get("channel_signal_grid"),
            sample_id=sample,
        )
        features = region_features_from_labels(
            artifacts["region_labels"]["blur"],
            artifacts["regions"]["blur"],
            float(factor),
            "Detected_Blur",
            simplify_px=float(config.get("boundary_simplify_px", 2.0)),
        )
        for feature in features:
            block = (feature.setdefault("properties", {}).setdefault("spacemap", {}))
            block["trial_name"] = name
            block["trial_parameters"] = {
                "blur_threshold_mode": config.get("blur_threshold_mode"),
                "blur_reference_mode": config.get("blur_reference_mode"),
                "blur_sensitivity_level": config.get("blur_sensitivity_level"),
                "blur_sensitivity": config.get("blur_sensitivity"),
                "blur_evidence_rule": config.get("blur_evidence_rule"),
                "marker_consensus_fraction": config.get("marker_consensus_fraction"),
                "blur_min_separation": config.get("blur_min_separation"),
            }
        trial_dir = trial_root / f"{position:02d}_{name}"
        trial_dir.mkdir()
        geojson_path = trial_dir / f"{sample}_detected_blur.geojson"
        write_feature_collection(features, geojson_path)
        regions = list(artifacts["regions"]["blur"])
        confidences = [float((item.get("spacemap") or {}).get("confidence") or 0.0) for item in regions]
        severities = [float((item.get("spacemap") or {}).get("severity") or 0.0) for item in regions]
        stats = dict(artifacts["stats"])
        row = {
            "trial": name,
            "trial_number": position,
            "geojson": str(geojson_path),
            "polygon_count": len(features),
            "blurry_percent": round(float(stats.get("blurry_percent") or 0.0), 6),
            "blur_population_found": bool(stats.get("blur_population_found")),
            "separation_gap": stats.get("blur_separation_gap"),
            "separation_share": stats.get("blur_separation_share"),
            "mean_confidence": round(float(np.mean(confidences)), 6) if confidences else 0.0,
            "mean_severity": round(float(np.mean(severities)), 6) if severities else 0.0,
            "blur_threshold_mode": stats.get("blur_threshold_mode"),
            "blur_threshold_used": stats.get("blur_threshold_used"),
            "sensitivity_level": config.get("blur_sensitivity_level"),
            "sensitivity": config.get("blur_sensitivity"),
            "evidence_rule": config.get("blur_evidence_rule"),
            "reference_mode": config.get("blur_reference_mode"),
            "marker_consensus_fraction": config.get("marker_consensus_fraction"),
            "min_separation": config.get("blur_min_separation"),
        }
        feature_sets.append(features)
        rows.append(row)
        trial_record = {
            "name": name,
            "parameters": config,
            "statistics": stats,
            "geojson": str(geojson_path),
            "regions": [item.get("spacemap") or {} for item in regions],
        }
        _atomic_json(trial_dir / "trial_report.json", trial_record)
        trial_records.append(trial_record)
        del artifacts
        gc.collect()

    emit("Comparing trials", "Measuring agreement and ranking candidates for visual review.")
    for row, stability in zip(rows, _stability_scores(feature_sets)):
        row["stability_iou"] = round(float(stability), 6)
        row["machine_screen_score"] = _machine_screen_score(row)
    rows.sort(key=lambda item: (float(item["machine_screen_score"]), -float(item["blurry_percent"])), reverse=True)
    comparison_path = destination / f"{sample}_blur_parameter_comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    candidate_rows = [item for item in rows if int(item.get("polygon_count") or 0) > 0]
    best = candidate_rows[0] if candidate_rows else None
    candidate_path: Path | None = None
    if best is not None:
        source_trial = Path(str(best["geojson"]))
        candidate_path = destination / f"{sample}_detected_blur_provisional.geojson"
        shutil.copy2(source_trial, candidate_path)
        best_record = next(item for item in trial_records if item["name"] == best["trial"])
        best_payload = {
            "status": "provisional_needs_visual_review",
            "warning": "The machine screen only ranks candidates. It is not a manual reference or final scientific approval.",
            "selected_trial": best["trial"],
            "machine_screen_score": best["machine_screen_score"],
            "parameters": best_record["parameters"],
            "statistics": best_record["statistics"],
            "candidate_geojson": str(candidate_path),
            "comparison_csv": str(comparison_path),
        }
        next_state = "visual_review_required"
    else:
        closest = max(rows, key=lambda item: float(item.get("separation_gap") or 0.0))
        best_payload = {
            "status": "no_candidates_detected",
            "warning": (
                "Every tested configuration returned zero blur polygons. No trial was selected, "
                "and no visual review was performed because there were no candidate polygons."
            ),
            "selected_trial": None,
            "machine_screen_score": None,
            "parameters": None,
            "statistics": None,
            "closest_population_trial": closest.get("trial"),
            "largest_separation_gap": closest.get("separation_gap"),
            "comparison_csv": str(comparison_path),
        }
        next_state = "no_candidates_detected"
    best_config_path = destination / f"{sample}_blur_provisional_config.json"
    _atomic_json(best_config_path, best_payload)
    run_manifest = {
        "schema": "spacemap.blur_parameter_search.v1",
        "sample_id": sample,
        "image_path": str(image),
        "tissue_boundary_path": str(boundary),
        "project_config_path": str(project_config_path or ""),
        "microns_per_pixel": float(tissue_config["microns_per_pixel"]),
        "markers": marker_names,
        "trial_count": len(rows),
        "selected_trial": best["trial"] if best is not None else None,
        "outputs": {
            "comparison_csv": str(comparison_path),
            "provisional_config": str(best_config_path),
            "provisional_geojson": str(candidate_path) if candidate_path is not None else "",
            "trial_directory": str(trial_root),
        },
        "next_state": next_state,
    }
    manifest_path = destination / "blur_run_manifest.json"
    _atomic_json(manifest_path, run_manifest)
    if best is not None:
        emit(
            "Parameter search complete",
            f"Provisional trial: {best['trial']}; visual review is still required.",
        )
    else:
        emit(
            "Parameter search complete",
            f"All {len(rows)} configurations returned zero polygons; no trial was selected.",
        )
    return {**run_manifest, "manifest_path": str(manifest_path), "comparison": rows}


def run_bright_parameter_search(
    image_path: str | Path,
    tissue_boundary_path: str | Path,
    output_dir: str | Path,
    *,
    project_config_path: str | Path | None = None,
    microns_per_pixel: float | None = None,
    trials: list[dict[str, Any]] | None = None,
    maximum_trials: int = 10,
    progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run real SPACEMAP bright-glare tests and save a visual-review package."""
    from spacemap.tissue_annotation_utils import (
        DEFAULT_ARTIFACT_CONFIG,
        DEFAULT_TISSUE_CONFIG,
        PyramidImageReader,
        detect_artifacts,
        detect_tissue_mask,
        load_geojson_mask,
        region_features_from_labels,
        split_he_and_codex_channels,
        write_feature_collection,
    )

    image = Path(image_path).resolve()
    boundary = Path(tissue_boundary_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    trial_root = destination / "trials"
    trial_root.mkdir()
    sample = _sample_id(image)
    emit = progress or (lambda _stage, _message: None)

    saved_tissue, saved_artifacts, selected_markers = _project_tab_settings(project_config_path)
    tissue_config = {**DEFAULT_TISSUE_CONFIG, **saved_tissue}
    if microns_per_pixel is not None and float(microns_per_pixel) > 0:
        tissue_config["microns_per_pixel"] = float(microns_per_pixel)
    tissue_config["detect_blood_clot"] = False
    base_artifact = {**DEFAULT_ARTIFACT_CONFIG, **saved_artifacts}
    base_artifact["detect_bright_artifacts"] = True
    base_artifact["detect_filamentous_lint"] = False
    base_artifact["write_review_manifest"] = False

    reader = PyramidImageReader(image)
    he_indices, codex_indices = split_he_and_codex_channels(reader.channel_names)
    if selected_markers:
        codex_indices = [
            index
            for index in codex_indices
            if selected_markers.get(str(reader.channel_names[index]), False)
        ]
    if not codex_indices:
        raise ValueError("No selected CODEX marker channels were found for bright-artifact detection.")
    marker_names = [str(reader.channel_names[index]) for index in codex_indices]

    emit("Reading image", f"Reading {len(codex_indices)} selected CODEX channels once.")
    last_read = -1

    def image_progress(done: int, total: int, name: str) -> None:
        nonlocal last_read
        if done == last_read:
            return
        last_read = done
        emit("Reading image", f"Channel {done}/{total}: {name}")

    maps = detect_tissue_mask(
        reader,
        tissue_config,
        he_indices=he_indices,
        codex_indices=codex_indices,
        progress=image_progress,
        need_codex_maps=True,
    )
    factor = int(tissue_config.get("blur_downsample", 2))
    tissue_mask = load_geojson_mask(boundary, tuple(maps["shape"]), float(factor))
    if not tissue_mask.any():
        raise ValueError("The tissue-boundary GeoJSON produced an empty mask.")

    requested_trials = list(trials or default_bright_trials(base_artifact))
    limit = min(20, max(1, int(maximum_trials)))
    requested_trials = requested_trials[:limit]
    feature_sets: list[list[dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    trial_records: list[dict[str, Any]] = []
    for position, trial in enumerate(requested_trials, start=1):
        name = str(trial.get("name") or f"trial_{position:02d}")
        overrides = dict(trial.get("overrides") or {})
        config = {**base_artifact, **overrides}
        emit("Testing bright settings", f"Trial {position}/{len(requested_trials)}: {name}")
        artifacts = detect_artifacts(
            tissue_mask=tissue_mask,
            codex_projection=maps["codex_projection"],
            he_gray=maps["he_gray"],
            codex_robust_max=maps["codex_robust_max"],
            channel_positive_count=maps["channel_positive_count"],
            n_codex_channels=len(codex_indices),
            config=config,
            microns_per_pixel=float(tissue_config["microns_per_pixel"]),
            downsample=factor,
            border_px=int(maps.get("border_px", 0)),
            marker_names=marker_names,
            channel_focus_grid=maps.get("channel_focus_grid"),
            channel_signal_grid=maps.get("channel_signal_grid"),
            sample_id=sample,
        )
        features = region_features_from_labels(
            artifacts["region_labels"]["bright"],
            artifacts["regions"]["bright"],
            float(factor),
            "Detected_Bright",
            simplify_px=float(config.get("boundary_simplify_px", 2.0)),
        )
        for feature in features:
            block = feature.setdefault("properties", {}).setdefault("spacemap", {})
            block["trial_name"] = name
            block["trial_parameters"] = {
                "bright_flat_mean_percentile": config.get("bright_flat_mean_percentile"),
                "bright_flat_texture_percentile": config.get("bright_flat_texture_percentile"),
                "use_breadth_signal": config.get("use_breadth_signal"),
                "bright_breadth_fraction": config.get("bright_breadth_fraction"),
                "min_area_microns2_bright": config.get("min_area_microns2_bright"),
            }
        trial_dir = trial_root / f"{position:02d}_{name}"
        trial_dir.mkdir()
        geojson_path = trial_dir / f"{sample}_detected_bright.geojson"
        write_feature_collection(features, geojson_path)
        regions = list(artifacts["regions"]["bright"])
        top_marker_values: list[float] = []
        for region in regions:
            candidates = list((region.get("spacemap") or {}).get("candidate_markers") or [])
            if candidates:
                top_marker_values.append(float(candidates[0].get("signal_z") or 0.0))
        stats = dict(artifacts["stats"])
        row = {
            "trial": name,
            "trial_number": position,
            "geojson": str(geojson_path),
            "polygon_count": len(features),
            "total_area_um2": round(
                sum(float((item.get("spacemap") or {}).get("area_um2") or 0.0) for item in regions),
                3,
            ),
            "bright_percent": round(float(stats.get("bright_percent") or 0.0), 6),
            "mean_top_marker_signal_z": (
                round(float(np.mean(top_marker_values)), 6) if top_marker_values else 0.0
            ),
            "brightness_percentile": config.get("bright_flat_mean_percentile"),
            "texture_percentile": config.get("bright_flat_texture_percentile"),
            "use_breadth_signal": bool(config.get("use_breadth_signal")),
            "breadth_fraction": config.get("bright_breadth_fraction"),
            "minimum_area_um2": config.get("min_area_microns2_bright"),
        }
        feature_sets.append(features)
        rows.append(row)
        trial_record = {
            "name": name,
            "parameters": config,
            "statistics": stats,
            "geojson": str(geojson_path),
            "regions": [item.get("spacemap") or {} for item in regions],
        }
        _atomic_json(trial_dir / "trial_report.json", trial_record)
        trial_records.append(trial_record)
        del artifacts
        gc.collect()

    emit("Comparing trials", "Measuring agreement and ranking bright candidates for visual review.")
    for row, stability in zip(rows, _stability_scores(feature_sets)):
        row["stability_iou"] = round(float(stability), 6)
        row["machine_screen_score"] = _bright_machine_screen_score(row)
    rows.sort(
        key=lambda item: (
            float(item["machine_screen_score"]),
            -float(item["bright_percent"]),
        ),
        reverse=True,
    )
    comparison_path = destination / f"{sample}_bright_parameter_comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    candidate_rows = [item for item in rows if int(item.get("polygon_count") or 0) > 0]
    best = candidate_rows[0] if candidate_rows else None
    candidate_path: Path | None = None
    if best is not None:
        candidate_path = destination / f"{sample}_detected_bright_provisional.geojson"
        shutil.copy2(Path(str(best["geojson"])), candidate_path)
        best_record = next(item for item in trial_records if item["name"] == best["trial"])
        best_payload = {
            "status": "provisional_needs_visual_review",
            "warning": "Machine ranking is not a biological or visual approval.",
            "selected_trial": best["trial"],
            "machine_screen_score": best["machine_screen_score"],
            "parameters": best_record["parameters"],
            "statistics": best_record["statistics"],
            "candidate_geojson": str(candidate_path),
            "comparison_csv": str(comparison_path),
        }
        next_state = "visual_review_required"
    else:
        best_payload = {
            "status": "no_candidates_detected",
            "warning": "Every tested configuration returned zero bright-artifact polygons.",
            "selected_trial": None,
            "machine_screen_score": None,
            "parameters": None,
            "statistics": None,
            "candidate_geojson": "",
            "comparison_csv": str(comparison_path),
        }
        next_state = "no_candidates_detected"
    best_config_path = destination / f"{sample}_bright_provisional_config.json"
    _atomic_json(best_config_path, best_payload)
    run_manifest = {
        "schema": "spacemap.bright_parameter_search.v1",
        "artifact_kind": "bright",
        "sample_id": sample,
        "image_path": str(image),
        "tissue_boundary_path": str(boundary),
        "project_config_path": str(project_config_path or ""),
        "microns_per_pixel": float(tissue_config["microns_per_pixel"]),
        "markers": marker_names,
        "trial_count": len(rows),
        "selected_trial": best["trial"] if best is not None else None,
        "outputs": {
            "comparison_csv": str(comparison_path),
            "provisional_config": str(best_config_path),
            "provisional_geojson": str(candidate_path) if candidate_path is not None else "",
            "trial_directory": str(trial_root),
        },
        "next_state": next_state,
    }
    manifest_path = destination / "bright_run_manifest.json"
    _atomic_json(manifest_path, run_manifest)
    if best is not None:
        emit(
            "Parameter search complete",
            f"Provisional bright trial: {best['trial']}; visual review is still required.",
        )
    else:
        emit(
            "Parameter search complete",
            f"All {len(rows)} configurations returned zero bright polygons; no trial was selected.",
        )
    return {**run_manifest, "manifest_path": str(manifest_path), "comparison": rows}
