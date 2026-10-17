"""Studio's Consensus ML (Tab 7): several methods' phenotypes combined; a model learns from the cells they agree on.

The settings are classic Tab 7's own file (``outputs/phenotyping/cache/consensus_ml_params.json``) with
classic's scientific defaults. The job runs the shared ``spacemap.ml_consensus`` engine
with a configurable CPU budget and an explicit optional GPU backend.
Its report and figures go into the job's folder; nothing in the project changes except four new
result columns, kept beside the table (pending, Parquet) until the user writes them in:

    <output>            each cell's phenotype
    <output>_score      the agreeing share (consensus cells) or the model's probability
    <output>_consensus  1 where the methods agreed
    <output>_refined    the phenotype, or "Others" below the score threshold

Replacing existing output columns requires the explicit overwrite option.
"""

import json
from pathlib import Path
import re
import subprocess
import sys
import time
import threading

import numpy as np
import pandas as pd

DEFAULTS = {
    "markers": [],
    "phenotyping_methods": [],
    "min_consensus_methods": 3,
    "exclude_phenotypes": ["Others"],
    "ml_method": "random_forest",
    "n_estimators": 1500,
    "cpu_workers": 0,
    "execution_backend": "cpu",
    "gpu_bins": 256,
    "test_size": 0.3,
    "random_state": 42,
    "min_training_cells": 50,
    "use_z_normalization": False,
    "batch_normalization_col": "batch_id",
    "enable_filter": True,
    "filter_col": "tissue",
    "filter_val": "1",
    "validation_ground_truth_col": "",
    "output_phenotype_col": "phenotype_ml_consensus",
    "refined_confidence_threshold": 0.75,
    "refined_low_confidence_label": "Others",
    "morphology_cols": [],
    "compartment_cols": [],
    "feature_weights": {"markers": 1.0, "morphology": 0.0, "compartments": 0.0},
    "figure_dpi": 300,
}
METHODS = ("random_forest", "gradient_boosting", "svm")
EDITABLE = {
    "markers": "list",
    "phenotyping_methods": "list",
    "min_consensus_methods": (int, 2, 20),
    "exclude_phenotypes": "names",
    "ml_method": METHODS,
    "n_estimators": (int, 10, 5000),
    "cpu_workers": (int, 0, 4096),
    "execution_backend": ("cpu", "gpu", "auto"),
    "gpu_bins": (int, 16, 4096),
    "use_z_normalization": bool,
    "enable_filter": bool,
    "validation_ground_truth_col": "column",
    "output_phenotype_col": "output",
    "refined_confidence_threshold": (float, 0.0, 1.0),
}
OUTPUT = re.compile(r"phenotype_[A-Za-z0-9_]{1,100}")
_HARDWARE = {}
_HARDWARE_LOCK = threading.Lock()


def hardware():
    """Bounded isolated CUDA probe; share its short-lived result across UI reads."""
    from spacemap.consensus_execution import cpu_budget
    with _HARDWARE_LOCK:
        if time.monotonic() - _HARDWARE.get("checked", -float('inf')) > 60:
            try:
                probe = subprocess.run([sys.executable, "-m", "spacemap.consensus_execution"],
                                       capture_output=True, text=True, timeout=12, check=True)
                gpu = json.loads(probe.stdout.strip().splitlines()[-1])
            except (subprocess.SubprocessError, ValueError, OSError, IndexError):
                gpu = {"available": False, "reason": "GPU check did not complete. CPU remains available."}
            _HARDWARE.update(checked=time.monotonic(), gpu=gpu)
        return {"cpu": cpu_budget(), "gpu": _HARDWARE["gpu"]}


def cache_path(root):
    return Path(root) / "outputs" / "phenotyping" / "cache" / "consensus_ml_params.json"


def values(root):
    path = cache_path(root)
    saved = json.loads(path.read_text()) if path.is_file() else {}
    saved = saved if isinstance(saved, dict) else {}
    return {**DEFAULTS, **{key: value for key, value in saved.items() if value is not None and key in DEFAULTS}}


def check(changes):
    from .leiden import COLUMN

    if not isinstance(changes, dict) or set(changes) - set(EDITABLE):
        raise ValueError("Unknown Consensus ML setting.")
    for key, value in changes.items():
        kind = EDITABLE[key]
        if kind == "list":
            if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value) or len(value) > 200:
                raise ValueError(f"Choose the {key.replace('_', ' ')}.")
        elif kind == "names":
            if not isinstance(value, list) or not all(isinstance(v, str) and 0 < len(v) <= 100 for v in value):
                raise ValueError("List the phenotypes that cannot form a consensus.")
        elif kind == "column":
            if not isinstance(value, str) or (value and not COLUMN.fullmatch(value)):
                raise ValueError("Choose a column, or none.")
        elif kind == "output":
            if not isinstance(value, str) or not OUTPUT.fullmatch(value):
                raise ValueError("Name the result phenotype_ followed by letters, numbers or underscores.")
        elif isinstance(kind, tuple) and isinstance(kind[0], str):
            if value not in kind:
                raise ValueError(f"Choose one of {', '.join(kind)}.")
        elif kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Choose on or off for {key}.")
        else:
            number, low, high = kind
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"{key.replace('_', ' ')} must be between {low} and {high}.")
            if number is int and int(value) != value:
                raise ValueError(f"{key.replace('_', ' ')} must be a whole number.")
    return changes


def save(root, changes, expected):
    """Merge *changes* into classic's Consensus ML settings file."""
    from .settings import project_lock, revision

    check(changes)
    with project_lock(root):
        if expected != revision(values(root)):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        path = cache_path(root)
        saved = json.loads(path.read_text()) if path.is_file() else {}
        saved.update(changes)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(path)
        return values(root)


def free_name(name, taken):
    """*name*, or its next version (_2, _3...) when a column already has it or its derived columns."""
    version, found = 1, name
    while any(column in taken for column in (found, found + "_score", found + "_consensus", found + "_refined")):
        version += 1
        found = f"{name}_{version}"
    return found


def check_run(chosen, available):
    check({key: chosen[key] for key in EDITABLE if key in chosen})
    if chosen.get("execution_backend", "cpu") != "cpu" and chosen["ml_method"] != "random_forest":
        raise ValueError("GPU execution is available for Random Forest only. Choose CPU for this classifier.")
    methods = chosen["phenotyping_methods"]
    if len(methods) < 2:
        raise ValueError("Choose at least two phenotyping methods to combine.")
    if not 2 <= int(chosen["min_consensus_methods"]) <= len(methods):
        raise ValueError(f"The agreement needed must be between 2 and {len(methods)} methods.")
    required = [*methods, *chosen["markers"], *(chosen.get("morphology_cols") or []),
                *(chosen.get("compartment_cols") or [])]
    if chosen["enable_filter"]:
        required.append(chosen["filter_col"])
    if chosen["use_z_normalization"]:
        required.append(chosen["batch_normalization_col"])
    if chosen["validation_ground_truth_col"]:
        required.append(chosen["validation_ground_truth_col"])
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"The table has no {', '.join(missing[:5])}.")
    if not chosen["markers"]:
        raise ValueError("Choose the markers the model learns from.")


def _summary(folder):
    """The engine's numbers from its report (ml_consensus_summary.txt)."""
    found = {}
    path = Path(folder) / "ml_consensus_summary.txt"
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"(Consensus cells|Consensus training cells|Validation accuracy|Balanced accuracy|Macro F1|Validation mode): (.+)",
            line,
        )
        if match:
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip()
            try:
                found[key] = float(value) if "." in value or value.isdigit() else value
            except ValueError:
                found[key] = value
    return found


def run(settings, folder, report):
    """The job: classic's Consensus ML on the combined table; four new pending columns."""
    from spacemap import dataio
    from spacemap.ml_consensus import ConsensusConfig, run_ml_consensus_analysis
    from spacemap.consensus_execution import resolve_execution

    from .cluster_profiles import build
    from .tasks import save_result, validate_table

    validate_table(settings)
    table, chosen = settings["table"], settings["config"]
    anchor = settings.get("cell_id_col", "cell_id")
    available = dataio.columns(table)
    check_run(chosen, available)
    report("Checking CPU / GPU resources…", 0.02)
    execution = resolve_execution(chosen.get("execution_backend", "cpu"),
                                  chosen.get("cpu_workers", 0), chosen.get("gpu_bins", 256))
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "execution.json").write_text(json.dumps(execution, indent=2))
    if execution.get("fallback_reason"):
        report(execution["fallback_reason"] + " Using CPU as requested by GPU if available.", None)
    if execution["backend"] == "gpu":
        report(f"GPU: {execution['gpu']['name']} · cuML {execution['gpu']['version']} · "
               f"{execution['gpu_bins']} split bins. GPU splits differ from CPU.", None)
    else:
        report(f"CPU: {execution['workers']} workers / {execution['available']} allocated logical CPUs "
               f"(requested: {execution['requested'] or 'Auto'}). Native threads: 1.", None)
    extra = [
        chosen["batch_normalization_col"] if chosen["use_z_normalization"] else None,
        chosen["filter_col"] if chosen["enable_filter"] else None,
        chosen["validation_ground_truth_col"] or None,
        "batch_id",
    ]
    wanted = list(
        dict.fromkeys(
            [anchor, *chosen["phenotyping_methods"], *chosen["markers"],
             *(chosen.get("morphology_cols") or []), *(chosen.get("compartment_cols") or []),
             *[c for c in extra if c and c in available]]
        )
    )
    report("Reading the combined table…", 0.03)
    frame = dataio.load(table, columns=wanted)
    output = settings["output_col"]
    config = ConsensusConfig(
        unified_csv=Path(table),
        output_dir=Path(folder) / "report",
        phenotyping_methods=list(chosen["phenotyping_methods"]),
        clustering_markers=list(chosen["markers"]),
        morphology_cols=list(chosen["morphology_cols"] or []),
        compartment_cols=list(chosen["compartment_cols"] or []),
        feature_weights=dict(chosen["feature_weights"]),
        use_z_normalization=bool(chosen["use_z_normalization"]),
        batch_normalization_col=chosen["batch_normalization_col"],
        min_consensus_methods=int(chosen["min_consensus_methods"]),
        test_size=float(chosen["test_size"]),
        n_estimators=int(chosen["n_estimators"]),
        cpu_workers=execution["workers"],
        execution_backend=execution["backend"],
        gpu_bins=execution["gpu_bins"],
        log_to_stdout=False,
        progress_interval=None,  # Jobs displays elapsed time independently; logs record stage changes.
        random_state=int(chosen["random_state"]),
        min_training_cells=int(chosen["min_training_cells"]),
        figure_dpi=int(chosen["figure_dpi"]),
        write_backup_csv=False,
        update_source_csv=False,  # Studio never writes the CSV
        excluded_phenotypes=[str(v) for v in chosen["exclude_phenotypes"]],
        output_phenotype_col=output,
        confidence_col_name=output + "_score",
        consensus_col_name=output + "_consensus",
        refined_confidence_threshold=float(chosen["refined_confidence_threshold"]),
        refined_low_confidence_label=str(chosen["refined_low_confidence_label"]),
        ml_method=chosen["ml_method"],
        validation_ground_truth_column=chosen["validation_ground_truth_col"] or None,
        analysis_filter_column=chosen["filter_col"]
        if chosen["enable_filter"]
        else None,
        analysis_filter_value=chosen["filter_val"] if chosen["enable_filter"] else None,
    )
    held, lines = {}, []

    def progress(message):
        text = str(message).strip()
        if text and not set(text) <= set("=- "):
            lines.append(text)
            step = re.search(r"Step (\d)/8", text)
            report(text[-200:], 0.1 + 0.1 * int(step.group(1)) if step else None)

    ok = run_ml_consensus_analysis(
        config, progress_callback=progress, preloaded_df=frame, results_callback=lambda df: held.update(frame=df)
    )
    if not ok or "frame" not in held:
        failed = next((line for line in reversed(lines) if "[ERROR]" in line), "Consensus ML stopped.")
        raise ValueError(re.sub(r"^\[\d\d:\d\d:\d\d\]\s*\[ERROR\]\s*", "", failed))
    measured = json.loads((Path(folder) / "report" / "execution.json").read_text())
    execution.update(stage_seconds=measured["stage_seconds"],
                     training_cells=measured["training_cells"], features=measured["features"])
    result = held["frame"]
    columns = {
        output: result[output],
        output + "_score": pd.to_numeric(result[output + "_score"], errors="coerce"),
        output + "_consensus": result[output + "_consensus"],
        output + "_refined": result[output + "_refined"],
    }
    saved = save_result(settings, columns, frame[anchor], report)
    report("Describing the phenotypes…", 0.95)
    labels = result[output].astype("string")
    numbers = frame[chosen["markers"]].apply(pd.to_numeric, errors="coerce")
    described = numbers.assign(_group=labels).dropna(subset=["_group"])
    clusters, profile = build(described, chosen["markers"], "_group", name=str, exclude_low=False,
                              label="Input marker values; no additional transformation")
    consensus = pd.to_numeric(result[output + "_consensus"], errors="coerce")
    return {
        **saved,
        "column": output,
        "phenotypes": [
            {"name": item["name"], "cells": item["cells"], "top_markers": item["top_markers"]} for item in clusters
        ],
        "profile": profile,
        "consensus_share": round(float(np.nanmean(consensus)), 4) if consensus.notna().any() else None,
        "report": _summary(Path(folder) / "report"),
        "methods": list(chosen["phenotyping_methods"]),
        "execution": execution,
    }
