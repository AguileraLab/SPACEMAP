"""Reusable strict-analysis cache for Tab 9 paper-style figures.

The expensive models are fitted once per input/configuration fingerprint.  All
compatible figure panels then read the same out-of-fold predictions and audit
tables, so changing figure styling never retrains a model.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from spacemap.grouped_validation import run_grouped_learning_curve

from .f1_strict_runner import (
    build_strict_validation_bundle,
    strict_grouped_config,
)


ANALYSIS_CACHE_VERSION = 2
TABLE_FILES = {
    "oof_predictions": "oof_predictions.csv",
    "fold_audit": "fold_leakage_audit.csv",
    "f1_weights": "fold_specific_f1_weights.csv",
    "consensus_training_counts": "consensus_training_counts.csv",
    "learning_curve": "annotation_learning_curve.csv",
}


def publication_analysis_fingerprint(
    input_csv: str | Path,
    settings: Mapping[str, Any],
    reference_columns: Sequence[str],
    excluded_labels: Sequence[str],
    confidence_thresholds: Sequence[float],
) -> tuple[str, dict[str, Any]]:
    """Fingerprint every value that can change fitted models or OOF rows."""
    path = Path(input_csv).expanduser().resolve()
    stat = path.stat()
    payload = {
        "cache_version": ANALYSIS_CACHE_VERSION,
        "input_csv": str(path),
        "input_size": int(stat.st_size),
        "input_modified_ns": int(stat.st_mtime_ns),
        "settings": dict(settings),
        "reference_columns": [str(value) for value in reference_columns],
        "excluded_labels": sorted(str(value) for value in excluded_labels),
        "confidence_thresholds": [float(value) for value in confidence_thresholds],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20], payload


def publication_cache_directory(
    output_dir: str | Path,
    fingerprint: str,
) -> Path:
    return Path(output_dir).expanduser() / "publication_validation_cache" / fingerprint


def cached_publication_analysis(
    output_dir: str | Path,
    fingerprint: str,
) -> bool:
    directory = publication_cache_directory(output_dir, fingerprint)
    return (directory / "manifest.json").is_file() and all(
        (directory / filename).is_file() for filename in TABLE_FILES.values()
    )


def load_publication_analysis(
    output_dir: str | Path,
    fingerprint: str,
) -> dict[str, Any]:
    directory = publication_cache_directory(output_dir, fingerprint)
    if not cached_publication_analysis(output_dir, fingerprint):
        raise FileNotFoundError(
            f"No complete publication-validation cache exists for {fingerprint}."
        )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in TABLE_FILES.items():
        try:
            tables[name] = pd.read_csv(directory / filename, low_memory=False)
        except pd.errors.EmptyDataError:
            tables[name] = pd.DataFrame(
                columns=manifest.get("table_columns", {}).get(name, [])
            )
    return {
        **tables,
        "manifest": manifest,
        "fingerprint": fingerprint,
        "cache_directory": str(directory),
    }


def save_publication_analysis(
    analysis: Mapping[str, Any],
    output_dir: str | Path,
    fingerprint: str,
    fingerprint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    directory = publication_cache_directory(output_dir, fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    for name, filename in TABLE_FILES.items():
        table = analysis.get(name)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"Publication analysis did not return {name} as a table.")
        table.to_csv(directory / filename, index=False)
    manifest = {
        "cache_version": ANALYSIS_CACHE_VERSION,
        "fingerprint": fingerprint,
        "completed_at": datetime.now().astimezone().isoformat(),
        "fingerprint_payload": dict(fingerprint_payload),
        "analysis_metadata": dict(analysis.get("metadata", {})),
        "rows": {
            name: int(len(analysis[name]))
            for name in TABLE_FILES
            if isinstance(analysis.get(name), pd.DataFrame)
        },
        "table_columns": {
            name: [str(column) for column in analysis[name].columns]
            for name in TABLE_FILES
            if isinstance(analysis.get(name), pd.DataFrame)
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return load_publication_analysis(output_dir, fingerprint)


def run_publication_analysis(
    frame: pd.DataFrame,
    *,
    settings: Mapping[str, Any],
    reference_columns: Sequence[str],
    excluded_labels: Sequence[str],
    method_display_names: Mapping[str, str],
    confidence_thresholds: Sequence[float],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fit strict grouped models and return the reusable tables."""
    references = list(dict.fromkeys(str(value) for value in reference_columns))
    annotators = tuple(references[:2]) if len(references) >= 2 else None
    bundle = build_strict_validation_bundle(
        frame,
        settings=settings,
        selected_reference_columns=references,
        excluded_labels=list(excluded_labels),
        method_display_names=method_display_names,
        remove_primary_overlap=False,
        confidence_thresholds=list(confidence_thresholds),
        annotator_columns=annotators,
        progress_callback=progress_callback,
    )
    strict = bundle["strict_validation"]
    learning_curve = pd.DataFrame()
    if bool(settings.get("run_learning_curve", False)):
        learning_config = strict_grouped_config(
            settings,
            references,
            excluded_labels,
            annotators,
        )
        learning_estimators = int(settings.get("learning_n_estimators", 500))
        learning_repeats = int(settings.get("learning_repeats", 10))
        if bool(settings.get("quick_mode", False)):
            learning_estimators = min(
                learning_estimators,
                int(settings.get("quick_n_estimators", 100)),
            )
            learning_repeats = min(learning_repeats, 3)
        learning_curve = run_grouped_learning_curve(
            frame,
            learning_config,
            target_cells_per_class=settings.get(
                "learning_targets", [5, 10, 20, 40, 80, 160]
            ),
            repeats=max(1, learning_repeats),
            n_estimators=max(1, learning_estimators),
            progress_callback=progress_callback,
        )
    return {
        "oof_predictions": strict["oof_predictions"],
        "fold_audit": strict["fold_audit"],
        "f1_weights": strict["f1_weights"],
        "consensus_training_counts": strict["consensus_training_counts"],
        "learning_curve": learning_curve,
        "metadata": bundle["metadata"],
    }
