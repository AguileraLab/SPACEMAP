"""Scientific work performed by Studio's isolated worker process."""

import hashlib
import json
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from spacemap import dataio, labels, pending_columns
from spacemap.viewer.sources import fingerprint


def source_revision(settings):
    paths = [Path(settings["table"]), labels.sidecar_path(settings["table"])]
    paths.extend(Path(item["file"]) for item in pending_columns.pending(settings["table"]))
    paths.extend(Path(settings[key]) for key in ("panel", "thresholds", "settings") if settings.get(key))
    if settings.get("result") == "tls_candidates":
        paths.extend(Path(source["path"]) for source in (settings.get("tissue_sources") or {}).values())
    return hashlib.sha256(
        json.dumps([(str(path), fingerprint(path) if path.exists() else None) for path in paths]).encode()
    ).hexdigest()


def overwrite_columns(settings):
    """Only the explicitly requested result outputs may replace saved values."""
    columns = settings.get("overwrite_columns") or []
    output = settings["output_col"]
    auxiliary = settings.get("auxiliary_columns") or []
    allowed_auxiliary = {output + suffix for suffix in ("_score", "_confidence", "_consensus", "_refined")}
    allowed_auxiliary.update(settings.get(key) for key in ("confidence_column", "refined_column"))
    if settings.get("result") == "tls_candidates":
        from .tls import outputs
        allowed_auxiliary = set(outputs(output)[1:])
    if not isinstance(auxiliary, list) or any(not isinstance(c, str) or c not in allowed_auxiliary for c in auxiliary):
        raise ValueError("Choose valid score and refined output columns.")
    permitted = {output, settings.get("phenotype_col"), *auxiliary}
    output_pattern = r"phenotype_[A-Za-z0-9_]{1,118}"
    if settings.get("result") == "tls_candidates":
        from .tls import PREFIX
        output_pattern = PREFIX.pattern
    if settings.get("result") == "neighbourhoods":
        from .neighbourhoods import OUTPUT, RESULT_OUTPUT
        output_pattern = RESULT_OUTPUT.pattern
        names = settings.get("names_col")
        if names is not None:
            if not isinstance(names, str) or not OUTPUT.fullmatch(names) or output != names + "_cn":
                raise ValueError("Choose matching neighbourhood result and name columns.")
            permitted.add(names)
    config = settings.get("config") or {}
    protected = {settings.get(key) for key in ("cell_id_col", "roi_column", "label_column", "reference_column")}
    protected.update(["cell_id", "label", "batch_id", *settings.get("markers", []),
                      *settings.get("morphology_cols", []), *settings.get("methods", []),
                      *config.get("compartment_cols", []), config.get("filter_col"), config.get("batch_normalization_col"),
                      *config.get("phenotyping_methods", []), config.get("parent_col"), config.get("validation_ground_truth_col")])
    chosen = settings.get("values") or {}
    protected.update(chosen.get(key) for key in ("phenotype_col", "batch_col", "roi_col"))
    protected.update([settings.get("clusters_col"), settings.get("image_col")])
    if (not isinstance(columns, list) or any(not isinstance(name, str) or name not in permitted
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}" if name in auxiliary else output_pattern, name) for name in columns)
            or set(columns).intersection(protected)):
        raise ValueError("Only the selected result outputs can be overwritten; input measurements and reference labels are protected.")
    return set(columns)


def validate_table(settings):
    table = settings["table"]
    available = dataio.columns(table)
    column = settings["output_col"]
    # Results are phenotype_ columns; neighbourhood results are neighborhood columns (Tab 8's name).
    if settings.get("result") == "tls_candidates":
        from .tls import outputs
        outputs(column)
    elif settings.get("result") == "neighbourhoods":
        from .neighbourhoods import RESULT_OUTPUT
        if not isinstance(column, str) or not RESULT_OUTPUT.fullmatch(column):
            raise ValueError("Choose a new neighborhood output column using letters, numbers and underscores.")
    elif not isinstance(column, str) or not re.fullmatch(r"phenotype_[A-Za-z0-9_]{1,118}", column):
        raise ValueError("Choose a new phenotype_ output column using letters, numbers and underscores.")
    replacing = overwrite_columns(settings)
    outputs = {column, *(settings.get("auxiliary_columns") or [])}
    if settings.get("result") == "neighbourhoods" and settings.get("names_col"):
        outputs.add(settings["names_col"])
    if (outputs & set(available)) - replacing:
        raise ValueError("That output column already exists. Choose a new name or tick Overwrite existing results.")
    anchor = settings.get("cell_id_col", "cell_id")
    if anchor not in available:
        raise ValueError(f"{Path(table).name} has no {anchor} column. Choose the combined output from Batch correction at the top of Phenotyping.")
    ids = dataio.load(table, columns=[anchor])[anchor]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.astype(str).duplicated().any():
        raise ValueError(f"{anchor} must identify each cell once: this table has missing or repeated IDs. Choose the combined output from Batch correction.")
    if not settings.get("markers") or not set(settings["markers"]) <= set(available):
        raise ValueError("Choose markers that have saved measurements in this table.")
    return available


@contextmanager
def _restore_on_failure(table, columns):
    """Keep replaced sidecars intact if any part of the result save fails.

    Called under the table write lock; copying only the small result sidecars
    avoids rewriting or copying the combined measurement table.
    """
    directory = pending_columns.store_dir(table)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [pending_columns._column_file(Path(table), name) for name in columns]
    paths.extend([directory / pending_columns.ANCHOR_FILE, labels.sidecar_path(table)])
    with tempfile.TemporaryDirectory(prefix=".result-backup-", dir=directory) as backup:
        copies = {}
        for index, path in enumerate(paths):
            copy = Path(backup) / str(index)
            copies[path] = copy if path.exists() else None
            if path.exists():
                shutil.copy2(path, copy)
        try:
            yield
        except Exception:
            for path, copy in copies.items():
                if copy is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(copy, path)
            raise


def save_result(settings, columns, anchor, report):
    replacing = overwrite_columns(settings)
    report("Saving the result…" if replacing else "Saving the new result column…", 0.98)
    protect_output = getattr(report, "protect_output", None)
    if protect_output is not None:
        protect_output()
    with labels.write_lock(settings["table"]):
        if source_revision(settings) != settings["revision"]:
            raise ValueError(
                "The source measurements or labels changed during this run. Its result was not applied; start again with the current data."
            )
        if any(column in dataio.columns(settings["table"]) and column not in replacing for column in columns):
            raise ValueError("A result column with this name appeared during the run. Earlier data was kept.")
        with _restore_on_failure(settings["table"], columns) if replacing or settings.get("result") == "tls_candidates" else nullcontext():
            pending_columns.save(
                settings["table"],
                columns,
                anchor_column=settings.get("cell_id_col", "cell_id"),
                anchor_values=anchor,
                source_note="SPACEMAP Studio background job",
            )
            if replacing:
                labels.clear_columns(settings["table"], list(replacing.intersection(columns)))
    return {
        "table": settings["table"],
        "column": settings["output_col"],
        "columns": list(columns),
        "cells": len(anchor),
        "cell_id_column": settings.get("cell_id_col", "cell_id"),
    }


def resolve_scores(frame, thresholds, markers, options, roi_column):
    """The same positivity and score pipeline as classic, without silent fallback."""
    from spacemap.resolve_live import Settings
    from spacemap_gui.tabs.tab7_phenotyping.resolve_clustering import (
        _apply_thresholds_from_table, _apply_resolve_without_thresholds, _apply_resolve_z1_positivity,
    )
    from spacemap_gui.tabs.tab7_phenotyping.common import compute_auto_thresholds, _THRESHOLD_MODE_TO_KEY
    config = Settings.from_config(options)
    mode = options.get("threshold_mode", "Predefined (CSV)")
    if mode == "Z1 score (no threshold files)":
        return _apply_resolve_z1_positivity(frame, markers, roi_column,
            cutoff=float(options.get("z1_cutoff", 1.5)), cofactor=config.tie_score_cofactor,
            input_space=options.get("input_space", "raw"))
    if mode == "None (No Thresholds)":
        return _apply_resolve_without_thresholds(frame, markers, roi_column, options, list(config.norm_order))
    if mode != "Predefined (CSV)":
        method = _THRESHOLD_MODE_TO_KEY.get(mode)
        if not method:
            raise ValueError("Unknown RESOLVE threshold method; no labels were written.")
        thresholds = compute_auto_thresholds(frame, markers, method, batch_col=roi_column, params={"percentile": options.get("threshold_percentile", 95), "manual": options.get("threshold_manual", 0)})
    else:
        from .resolve_setup import threshold_frame
        thresholds = threshold_frame(thresholds.to_dict("records"), markers, frame[roi_column].astype(str))
    return _apply_thresholds_from_table(frame, thresholds, markers, roi_column, config.norm_flags,
        list(config.norm_order), "resolve", config.tie_score_transform, config.tie_score_cofactor)


def resolve_batch(frame, thresholds, markers, panel, options, roi_column):
    """Call classic's positivity path, including hierarchy and all tie-break rules."""
    from spacemap.ph_thres_utils import (
        assign_phenotypes_binary_tiebreak,
        apply_supportive_markers,
        apply_exclusion_markers,
    )
    from spacemap.resolve_live import hierarchy

    good = np.isfinite(frame[markers].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    result = pd.Series(None, index=frame.index, dtype=object)
    valid = frame.loc[good].copy()
    if valid.empty:
        return result
    binary, norm = resolve_scores(valid, thresholds, markers, options, roi_column)
    parents = hierarchy(panel.definitions)
    if (options.get("threshold_mode") in {"None (No Thresholds)", "Z1 score (no threshold files)"}
            and options.get("adaptive_hierarchy", True) and any(parents.values())):
        from spacemap_gui.tabs.tab7_phenotyping.resolve_clustering import _adaptive_resolve_hierarchy
        binary = _adaptive_resolve_hierarchy(valid, binary, panel.definitions, parents)
    both = pd.concat([binary, norm], axis=1)
    assigned = assign_phenotypes_binary_tiebreak(
        both,
        list(binary.columns),
        list(norm.columns),
        "studio_phenotype",
        panel.definitions,
        parents,
    )["studio_phenotype"]
    if panel.supportive:
        assigned = apply_supportive_markers(
            assigned, both, list(binary.columns), list(norm.columns), panel.definitions, panel.supportive
        )
    if panel.exclusions:
        assigned = apply_exclusion_markers(
            assigned, both, list(norm.columns), panel.definitions, panel.exclusions, reassign=True
        )
    result.loc[valid.index] = assigned.replace({"unclassified": options.get("unclassified_label", "Others")})
    return result


def resolve_all(settings, folder, report):
    from spacemap.resolve_live import Panel
    from spacemap.viewer.consensus_model import selected_values

    validate_table(settings)
    table = settings["table"]
    markers = settings["markers"]
    roi_column = settings.get("roi_column", "batch_id")
    options = settings.get("options") or {}
    identity = dataio.load(table, columns=list(dict.fromkeys([roi_column, settings.get("cell_id_col", "cell_id"),
        *([options["filter_col"]] if options.get("enable_filter") else [])]))).reset_index(drop=True)
    anchor = identity[settings.get("cell_id_col", "cell_id")]
    if anchor.isna().any() or anchor.astype(str).duplicated().any():
        raise ValueError("Cell IDs must be present and unique before saving a result.")
    if identity[roi_column].isna().any():
        raise ValueError("Some cells have no ROI name. Fill it in before running all ROIs.")
    panel = Panel.from_payload(json.loads(Path(settings["panel"]).read_text()))
    from .resolve_setup import filtered
    eligible = filtered(identity, options).index
    if not len(eligible):
        raise ValueError("No cells match the filter. No result was saved.")
    names = identity[roi_column].astype(str)
    thresholds = pd.DataFrame()
    if options.get("threshold_mode", "Predefined (CSV)") == "Predefined (CSV)":
        from .resolve_setup import threshold_frame
        thresholds = threshold_frame(pd.read_csv(settings["thresholds"]).to_dict("records"), markers, names.loc[eligible])
        thresholds.to_csv(Path(folder) / "resolve_used_thresholds.csv", index=False)
    output = np.full(len(identity), None, dtype=object)
    groups = names.loc[eligible].groupby(names.loc[eligible], sort=False).groups
    computed = []
    from spacemap.resolve_live import hierarchy
    adaptive = (options.get("threshold_mode") in {"None (No Thresholds)", "Z1 score (no threshold files)"}
                and options.get("adaptive_hierarchy", True) and any(hierarchy(panel.definitions).values()))
    if adaptive:
        # Classic's distinguishing-marker cutoff uses all parent candidates,
        # even though tie-score normalisation remains within each ROI.
        report("RESOLVE: reading parent/child marker patterns across all selected ROIs…", .1)
        raw = selected_values(settings, eligible.to_numpy(), dtype=np.float64)
        frame = pd.DataFrame(raw, columns=markers)
        frame[roi_column] = names.loc[eligible].to_numpy()
        output[eligible] = resolve_batch(frame, thresholds, markers, panel, options, roi_column).to_numpy()
    for index, (roi, positions) in enumerate([] if adaptive else groups.items()):
        report(f"RESOLVE: {roi} ({index + 1} of {len(groups)} ROIs)", 0.05 + 0.88 * index / len(groups))
        raw = selected_values(
            {**settings, "label_column": settings.get("label_column", "label")}, positions, dtype=np.float64
        )
        frame = pd.DataFrame(raw, columns=markers)
        frame[roi_column] = roi
        used_thresholds, run_options = thresholds, options
        mode = options.get("threshold_mode", "Predefined (CSV)")
        if mode not in {"Predefined (CSV)", "Z1 score (no threshold files)", "None (No Thresholds)"}:
            from spacemap_gui.tabs.tab7_phenotyping.common import compute_auto_thresholds, _THRESHOLD_MODE_TO_KEY
            good = np.isfinite(frame[markers].to_numpy(float)).all(axis=1)
            if not good.any():
                continue
            used_thresholds = compute_auto_thresholds(frame.loc[good], markers, _THRESHOLD_MODE_TO_KEY[mode], roi_column, params={"percentile": options.get("threshold_percentile", 95), "manual": options.get("threshold_manual", 0)})
            computed.append(used_thresholds)
            run_options = {**options, "threshold_mode": "Predefined (CSV)"}
        output[positions] = resolve_batch(
            frame, used_thresholds, markers, panel, run_options, roi_column
        ).to_numpy()
    if computed:
        pd.concat(computed, ignore_index=True).to_csv(Path(folder) / "resolve_used_thresholds.csv", index=False)
    result = save_result(settings, {settings["output_col"]: output}, anchor, report)
    result["rois"] = len(groups)
    result["roi_names"] = list(groups)
    result["left_out"] = int(pd.isna(output).sum())
    result["counts"] = pd.Series(output).value_counts().to_dict()
    result["threshold_mode"] = options.get("threshold_mode", "Predefined (CSV)")
    result["input_space"] = options.get("input_space", "raw")
    result["adaptive_hierarchy"] = adaptive
    return result


def validate_mr_ml(settings):
    """Reject name collisions and invalid thresholds before an expensive fit."""
    available = validate_table(settings)
    column = settings["output_col"]
    score = settings.get("confidence_column", column + "_confidence")
    refined = settings.get("refined_column", column + "_refined")
    if not isinstance(score, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", score):
        raise ValueError("Use letters, numbers and underscores for the score column.")
    if not isinstance(refined, str) or not re.fullmatch(r"phenotype_[A-Za-z0-9_]{1,118}", refined):
        raise ValueError("Choose a phenotype_ name for the filtered labels column.")
    if len({column, score, refined}) != 3:
        raise ValueError("Prediction, score and filtered label columns must have different names.")
    if ({score, refined} & set(available)) - overwrite_columns(settings):
        raise ValueError("An output column already exists. Choose new names or tick Overwrite existing results.")
    threshold = float(settings.get("confidence_threshold", 0.75))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The confidence threshold must be between 0 and 1.")
    if not str(settings.get("low_confidence_label", "Others")).strip():
        raise ValueError("Enter the label to use below the confidence threshold.")
    morphology = settings.get("morphology_cols", [])
    protected = {settings.get(key) for key in ("reference_column", "cell_id_col", "roi_column", "label_column")}
    protected.update(settings.get("markers", []))
    protected.update(("cell_id", "label", "batch_id"))
    if set(morphology) - set(available) or set(morphology) & protected or any(name.startswith("phenotype_") for name in morphology):
        raise ValueError("Choose additional measurement columns, excluding markers, cell IDs and phenotype labels.")
    return score, refined


def mr_ml(settings, folder, report):
    from spacemap.ml_mr_utils import run_ml_mr_analysis

    score, refined = validate_mr_ml(settings)
    config = settings.get('config') or {}
    if config.get('workflow') == 'saved_model':
        from .mr_ml import predict_saved
        return predict_saved(settings, folder, report)
    table = settings["table"]
    column = settings["output_col"]
    reference = settings["reference_column"]
    available = dataio.columns(table)
    if reference not in available:
        raise ValueError("Choose an existing manual reference column.")
    methods = list(dict.fromkeys(settings.get("methods", [])))
    if reference in methods or set(methods) - set(available):
        raise ValueError("Additional methods must be existing columns other than the training reference.")
    wanted = list(
        dict.fromkeys(
            [
                settings.get("cell_id_col", "cell_id"),
                settings.get("roi_column", "batch_id"),
                reference,
                *settings["markers"],
                *settings.get("morphology_cols", []),
                *config.get('compartment_cols', []),
                *([config['validation_ground_truth_col']] if config.get('validation_ground_truth_col') else []),
                *([config['batch_normalization_col']] if config.get('use_z_normalization') else []),
                *([config['filter_col']] if config.get('enable_filter') else []),
                *methods,
            ]
        )
    )
    frame = dataio.load(table, columns=wanted)
    anchor = frame[settings.get("cell_id_col", "cell_id")]
    if anchor.isna().any() or anchor.astype(str).duplicated().any():
        raise ValueError("Cell IDs must be present and unique.")
    validation = config.get('validation_ground_truth_col')
    if validation and validation != reference and config.get('drop_overlapping_validation', True):
        from spacemap.ml_mr_utils import _valid_reference_mask
        overlap = (_valid_reference_mask(frame[reference], settings.get('excluded_phenotypes', ['Others'])) &
                   _valid_reference_mask(frame[validation], settings.get('excluded_phenotypes', ['Others'])))
        if overlap.any():
            frame.loc[overlap, validation] = np.nan
            report(f'Leaving {int(overlap.sum()):,} training-reference cells out of external validation.')
    result_frames = []
    failures = []

    def progress(message):
        report(message)
        if "[ERROR]" in message:
            failures.append(message.split("[ERROR]", 1)[1].strip())

    report("Training from the selected manual reference column…", 0.1)
    okay = run_ml_mr_analysis(
        Path(table),
        None,
        Path(folder) / "results",
        reference,
        methods,
        marker_suffix="",
        clustering_markers=settings["markers"],
        morphology_cols=settings.get("morphology_cols", []),
        compartment_cols=config.get('compartment_cols', []),
        feature_weights=config.get('feature_weights'),
        use_z_normalization=config.get('use_z_normalization', False),
        batch_normalization_col=config.get('batch_normalization_col', 'batch_id'),
        test_size=config.get('test_size', .3),
        n_estimators=config.get('n_estimators', 150),
        random_state=config.get('random_state', 42),
        ml_method=config.get('ml_method', settings.get("method", "random_forest")),
        validation_ground_truth_column=config.get('validation_ground_truth_col') or None,
        analysis_filter_column=config.get('filter_col') if config.get('enable_filter') else None,
        analysis_filter_value=config.get('filter_val'),
        output_phenotype_col=column,
        confidence_col_name=score,
        refined_output_phenotype_col=refined,
        refined_confidence_threshold=settings.get("confidence_threshold", 0.75),
        refined_low_confidence_label=settings.get("low_confidence_label", "Others"),
        excluded_phenotypes=settings.get("excluded_phenotypes", ["Others"]),
        preloaded_df=frame,
        progress_callback=progress,
        results_callback=result_frames.append,
        write_backup_csv=False,
        update_source_csv=False,
        log_to_stdout=False,
    )
    if not okay or not result_frames:
        raise ValueError(failures[-1] if failures else "MR-ML did not produce a result. Read the job log for the missing input.")
    result = result_frames[-1]
    if not result.index.equals(frame.index):
        raise ValueError("The prediction row order changed. No result was saved.")
    columns = {name: result[name] for name in (column, score, refined)}
    response = save_result(settings, columns, anchor, report)
    # Show the conservative result by default; both label columns remain available.
    response.update(column=refined, prediction_column=column, confidence_column=score,
                    reference_only=not methods, scored_cells=int(result[score].notna().sum()))
    response["features"] = {"markers": settings["markers"], "morphology": settings.get("morphology_cols", [])}
    comparison = {}
    for method in methods:
        both = result[column].notna() & frame[method].notna()
        if both.any():
            names = frame.loc[both, method].astype(str)
            same = result.loc[both, column].astype(str) == names
            comparison[method] = {
                "cells": int(both.sum()),
                "agreement": float(same.mean()),
                "per_phenotype": {
                    str(phenotype): {
                        "cells": int((names == phenotype).sum()),
                        "agreement": float(same[names == phenotype].mean()),
                    }
                    for phenotype in sorted(names.unique())
                },
            }
    response["agreement"] = comparison
    weights_file = Path(folder) / 'results/training_reference_f1_weights.csv'
    if weights_file.exists() and methods:
        weights = pd.read_csv(weights_file, index_col=0).fillna(0)
        response['reference_f1'] = {str(name): {str(method): float(value) for method, value in row.items()}
                                    for name, row in weights.iterrows()}
    return response


def train_cellpose(settings, folder, report):
    from .training import train

    return train(settings, folder, report)


def som(settings, folder, report):
    from .clustering import run

    return run(settings, folder, report)


def subphenotype(settings, folder, report):
    from .subphenotypes import run

    return run(settings, folder, report)


def consensus_ml(settings, folder, report):
    from .consensus import run

    return run(settings, folder, report)


def export_table(settings, folder, report):
    from .saved import export

    return export(settings, folder, report)


def vortex(settings, folder, report):
    from .clustering import run

    return run({**settings, "method": "vortex"}, folder, report)


def correct_batches(settings, folder, report):
    from .correction import run

    return run(settings, folder, report)


def score_cellpose(settings, folder, report):
    from .training import score_versions

    return score_versions(settings, folder, report)


def export_background(settings, folder, report):
    """Each listed image's thresholds (CSV) and/or background-removed image (OME-TIFF)."""
    from .background import export

    for item in settings["images"]:
        if not Path(item["image"]).is_file():
            raise ValueError(f"The image {Path(item['image']).name} is no longer there.")
    done = export(settings["images"], report)
    return {"images": done, "count": len(done)}


def detect_background(settings, folder, report):
    """Auto background levels for each listed image, saved as its threshold table."""
    from .background import detect

    found = detect(settings["images"], report)
    return {"images": found, "count": len(found)}


def remove_background(settings, folder, report):
    """Background removed from each listed image (blank cycles and/or levels), with its chosen markers."""
    from .background import remove

    for item in settings["images"]:
        if not Path(item["image"]).is_file():
            raise ValueError(f"The image {Path(item['image']).name} is no longer there.")
    written = remove(settings["images"], report)
    return {"images": written, "count": len(written)}


def cell_tagging(settings, folder, report):
    """Tab 5's spatial cell tagging for the listed cell tables."""
    from .tagging import run

    return run(settings, folder, report)


def cell_interactions(settings, folder, report):
    from .interactions import run

    return run(settings, folder, report)


def neighbourhoods(settings, folder, report):
    from .neighbourhoods import run

    return run(settings, folder, report)


def leiden(settings, folder, report):
    from .leiden import run

    return run(settings, folder, report)


def name_clusters(settings, folder, report):
    from .leiden import name_clusters as save_names

    return save_names(settings, folder, report)


def train_reference_model(settings, folder, report):
    from .reference_models import train

    try:
        return train(settings, folder, report)
    except ValueError as exc:
        if settings.get("auto_train") and ("two phenotypes" in str(exc) or "five complete examples" in str(exc)):
            return {"table":str(settings["table"]), "auto_train":True, "models":[], "waiting":str(exc)}
        raise


def subtract_background(settings, folder, report):
    """Tab 2's background removal for each listed image and its saved levels."""
    from . import background

    images = settings["images"]
    for item in images:
        if not Path(item["image"]).is_file():
            raise ValueError(f"The image {Path(item['image']).name} is no longer there.")
        if not Path(item["table"]).is_file():
            raise ValueError(f"Save background levels for {Path(item['image']).name} first.")
    written = background.subtract(images, report)
    return {"images": written, "count": len(written)}


def extract_rois(settings, folder, report):
    """Tab 3's ROI extraction for each listed image and its ROI file."""
    from . import rois

    for item in settings["images"]:
        if not Path(item["image"]).is_file():
            raise ValueError(f"The image {Path(item['image']).name} is no longer there.")
        if not rois.roi_names(item["rois"]):
            raise ValueError(f"Draw ROIs on {Path(item['image']).name} first.")
    written = rois.extract(settings["images"], report)
    return {"images": written, "count": len(written)}


def export_annotations(settings, folder, report):
    """The Annotation step's Export for each listed image: the project's export rules (Export, Subtract, Keep inside)
    applied to its annotations, one GeoJSON named <prefix><image name><suffix>.geojson in the export folder. An image
    without any annotation ticked Export, or without the Keep inside one, is listed as not exported."""
    from . import annotation, annotation_names, annotation_workflow
    from .settings import read_config

    root = Path(settings["root"])
    config = read_config(root)
    output = annotation.all_folders(root, config.get("directory_mappings", {}))["output"]
    naming = annotation_names.export_settings(root, config)
    done, failed = [], []
    for number, image in enumerate(settings["images"]):
        name = Path(image).name
        report(f"{name}: exporting ({number + 1} of {len(settings['images'])})", number / len(settings["images"]))
        try:
            result = annotation_names.export(root, config, image, output, settings.get("merge", True), naming)
            annotation_workflow.record(root, image=image, target=result["path"], event={"operation": "export_annotations",
                                       "parameters": {"names": result.get("names"), "merge_same_names": settings.get("merge", True),
                                                      "layers": result.get("sources")}})
            done.append({"image": name, "file": result["filename"], "count": result["count"], "names": result["names"], "left_out": result.get("left_out", [])})
            said = f"{name}: {', '.join(result['names'])} in {result['filename']}"
            if result.get("small_pieces_dropped"):
                said += f", {result['small_pieces_dropped']} small piece{'s' if result['small_pieces_dropped'] > 1 else ''} dropped"
            if result.get("left_out"):
                said += f" (nothing left of {', '.join(result['left_out'])}" + (f" inside {result['inside']})" if result.get("inside") else ")")
            report(said, (number + 1) / len(settings["images"]))
        except (ValueError, OSError) as error:
            failed.append({"image": name, "error": str(error)})
            report(f"{name}: not exported: {error}", (number + 1) / len(settings["images"]))
    if not done and failed:
        raise ValueError(failed[0]["image"] + ": " + failed[0]["error"])
    return {"images": done, "failed": failed, "count": len(done), "folder": str(annotation_names.export_folder(root, naming, config))}


def import_annotations(settings, folder, report):
    """Every GeoJSON in a folder imported for the image it names (Annotation, Segmentation: Import a folder)."""
    from . import annotation_names
    from .settings import read_config

    root = Path(settings["root"])
    config = read_config(root)
    config = {**config, "directory_mappings": {**(settings.get("mappings") or {}), **(config.get("directory_mappings") or {})}}
    return annotation_names.import_directory(root, config, settings["folder"], report)


def classify_images(settings, folder, report):
    """Classification for each listed image with the same settings, inside its own Work within area. Each finished
    image keeps its run (class map and review), so it opens with its classes while the others are still running."""
    from . import annotation, annotation_inputs, classification
    from .settings import read_config

    root = Path(settings["root"])
    config = read_config(root)
    output = annotation.all_folders(root, config.get("directory_mappings", {}))["output"]
    done, failed = [], []
    for number, image in enumerate(settings["images"]):
        name = Path(image).name
        report(f"{name}: classifying ({number + 1} of {len(settings['images'])})", number / len(settings["images"]))
        try:
            scope = annotation_inputs.scope_path(root, image, output, "classes")
            prepared, found, chosen, seconds = classification.compute(image, config, scope, settings["options"])
            classification.persist(root, image, prepared, found)
            target = classification.directory(root, image) / (found["run_id"] + ".json")
            saved = classification.read_json(target, {})
            state = classification.review(saved.get("review"), np.flatnonzero(found["counts"]).tolist())
            classification.save_review(root, image, found, chosen, state)
            classes = int((found["counts"] > 0).sum())
            done.append({"image": name, "classes": classes, "seconds": round(seconds, 1)})
            report(f"{name}: {classes} classes ({seconds:.0f} s)", (number + 1) / len(settings["images"]))
        except (ValueError, OSError, RuntimeError) as error:
            failed.append({"image": name, "error": str(error)})
            report(f"{name}: not classified: {error}", (number + 1) / len(settings["images"]))
    if not done and failed:
        raise ValueError(failed[0]["image"] + ": " + failed[0]["error"])
    return {"images": done, "failed": failed, "count": len(done)}


def marker_regions(settings, folder, report):
    """Marker regions for each listed image, saved as its regions file."""
    from . import regions

    for item in settings["images"]:
        if not Path(item["image"]).is_file():
            raise ValueError(f"The image {Path(item['image']).name} is no longer there.")
    found = regions.detect_many(settings["images"], report)
    return {"images": found, "count": sum(item["regions"] for item in found)}


def detect_roi_cores(settings, folder, report):
    """Classic Tab 3's TMA core detection for each listed image, saved as its ROIs."""
    from . import rois

    for item in settings["images"]:
        if not Path(item["image"]).is_file():
            raise ValueError(f"The image {Path(item['image']).name} is no longer there.")
    found = rois.detect_many(settings["images"], report)
    return {"images": found, "count": sum(item["cores"] for item in found)}


def segment_images(settings, folder, report):
    """Tab 5's Cellpose segmentation for each listed image."""
    from . import segmentation

    for item in settings["images"]:
        if not Path(item["image"]).is_file():
            raise ValueError(f"The image {Path(item['image']).name} is no longer there.")
    # Each image's area (its polygons of the chosen name) is cut out here, so the button returns at once; an image
    # whose area cannot be made is left out and said, the others go on.
    pending = [item for item in settings["images"] if (item.get("area") or {}).get("pending") is not None]
    skipped = []
    if pending:
        from .settings import read_config

        root = Path(settings["root"])
        config = read_config(root)
        for number, item in enumerate(pending):
            name = Path(item["image"]).name
            report(f"{name}: preparing its area ({number + 1} of {len(pending)})", 0)
            try:
                item["area"] = {"name": item["area"]["name"],
                                "path": segmentation.area_path(root, config, settings.get("mappings") or {}, name, item["area"]["pending"])}
            except (ValueError, OSError) as error:
                skipped.append({"image": name, "error": str(error)})
                report(f"{name}: not segmented: {error}", 0)
        settings = {**settings, "images": [item for item in settings["images"] if (item.get("area") or {}).get("pending") is None]}
        if not settings["images"]:
            raise ValueError("No image could be prepared: " + "; ".join(f"{s['image']} ({s['error']})" for s in skipped[:3]))
    results = segmentation.run(settings["images"], settings["values"], report)
    found = {"images": results, "count": len(results), "cells": sum(item["cells"] for item in results), "skipped": skipped}
    if settings.get("tag"):
        found["tagging"] = _tag_after_segmenting(settings["tag"], settings["images"], folder, report)
    return found


def _tag_after_segmenting(root, items, folder, report):
    """Then tag: the Cell tagging step, with its own settings (columns, names, units, pieces, beds), on the tables just
    written. Says why when there is nothing to tag, without failing the segmentation."""
    from spacemap_gui.tabs import cell_tagging_ui as tab5

    from . import segmentation, tagging
    from .settings import read_config

    state = tagging.values(root, read_config(root))
    output = Path(items[0]["outputs"])
    state[tab5.CSV_DIR_KEY] = str(output)  # the tables Segmentation just wrote, wherever Cell tagging's folder points
    state[tab5.MODE_KEY] = tab5.MODE_PER_IMAGE
    rules, edge, *_ = tab5.build_engine_inputs(state)
    if not any(rule.is_active() for rule in rules) and edge is None and not tab5.selected_bed_labels(state=state):
        report("Tagging skipped: no columns are chosen in the Cell tagging step yet.", 1.0)
        return {"skipped": "No columns chosen in Cell tagging."}
    found = tagging.sources(root, state)
    tables = [segmentation.results(output, Path(item["image"]).name)[0].name for item in items]
    tables = [name for name in tables if name in found.get("tables", [])]
    if not tables:
        report("Tagging skipped: the new cell tables were not found.", 1.0)
        return {"skipped": "The new cell tables were not found."}
    exports, _ = tagging._exports_and_images(root)
    report(f"Tagging {len(tables)} new table{'s' if len(tables) != 1 else ''} with the Cell tagging settings", 0.99)
    return tagging.run({"state": state, "found": found, "tables": tables, "root": str(root), "exports": exports}, folder, report)


def register_pairs(settings, folder, report):
    """Tab 1's registration for each listed CODEX and H&E pair."""
    from . import registration

    for item in settings["pairs"]:
        for path in [item["codex"], item["he"], *(item.get("extra") or [])]:
            if not Path(path).is_file():
                raise ValueError(f"{Path(path).name} is no longer there.")
    written = registration.register(settings["pairs"], settings["options"], report)
    return {"images": written, "count": len(written)}


def detect_tissue(settings, folder, report):
    """Tab 4's tissue detection for each listed image, writing its annotation files."""
    from . import annotation

    for item in settings["images"]:
        if not Path(item["image"]).is_file():
            raise ValueError(f"The image {Path(item['image']).name} is no longer there.")
    written = annotation.run(settings["images"], settings["config"], settings["values"], report)
    if settings.get('root'):
        from . import annotation_inputs
        for item in settings['images']:
            annotation_inputs.save_selection(settings['root'], settings['config'], item['image'], item['output'],
                                             'tissue', {'mode':'tissue','class_name':''})
    return {"files": written, "count": len(settings["images"])}


def clean_tissue(settings, folder, report):
    """Compatibility command: detect then subtract, retaining current tissue and its history."""
    from . import annotation, annotation_edits, artifacts

    def work(item):
        tissue = Path(item.get("tissue_path") or annotation.valid_tissue(item["output"], item["sample"]))
        expected = item.get("tissue_revision", annotation_edits.revision(tissue))
        if not tissue.is_file():
            annotation.detect(item["image"], settings["config"], settings["values"], output=item["output"],
                              sample=item["sample"], expected=expected)
            expected = annotation_edits.revision(tissue)
        artifact = annotation.drawn_artifacts(item["output"], item["sample"])
        artifacts.detect_editable(item["image"], settings["config"], item["output"], item["sample"],
                                  item.get("artifact_revision", annotation_edits.revision(artifact)), lambda text: report(text, None))
        before = sum(annotation_edits.shapes.area_px(f) for f in annotation_edits.read(tissue)[0])
        if not annotation_edits.read(artifact)[0]:
            return {"sample": item["sample"], "valid_percent": 100, "artifact_percent": 0}
        changed = annotation_edits.subtract(tissue, [artifact], expected, label=annotation.tissue_label(settings["config"]),
            kind=item.get('tissue_kind','tissue'), class_name=item.get('class_name',''))
        kept = sum(annotation_edits.shapes.area_px(f) for f in changed["features"])
        share = 100 * kept / max(before, 1)
        return {"sample": item["sample"], "valid_percent": share, "artifact_percent": 100 - share}
    return _each_image(settings, report, work)


def detect_artifacts(settings, folder, report):
    from . import artifacts

    return _each_image(settings, report, lambda item: artifacts.detect_editable(
        item["image"], settings["config"], item["output"], item["sample"], item["artifact_revision"],
        lambda text: report(text, None)))


def subtract_artifacts(settings, folder, report):
    from . import annotation, annotation_edits, artifact_learning

    def work(item):
        tissue = Path(item.get('tissue_path') or annotation.valid_tissue(item["output"], item["sample"]))
        artifact = annotation.drawn_artifacts(item["output"], item["sample"])
        # Refuse to subtract a different artifact set than the one the user reviewed.
        if annotation_edits.revision(artifact) != item["artifact_revision"]:
            raise ValueError("Artifacts changed while this job was waiting. Review them and subtract again.")
        reviewed = []
        classes = []
        for part, expected in item.get("reviewed_artifact_revisions", {}).items():
            path = artifact_learning.polygons_path(item["output"], item["sample"], part)
            if annotation_edits.revision(path) != expected:
                raise ValueError("Reviewed artifacts changed. Review them and subtract again.")
            if path.is_file():
                reviewed.append(path); classes.append(artifact_learning.CLASSES[part])
                if part == 'debris':
                    classes.extend(['Bright_spots_and_lint', *(artifact_learning.CLASSES[p] for p in ('blur', 'bright', 'lint'))])
        result = annotation_edits.subtract(tissue, [artifact, *reviewed], item["tissue_revision"],
            kind=item.get('tissue_kind','tissue'), class_name=item.get('class_name',''),
            overrides={str(artifact): classes}, label=annotation.tissue_label(settings["config"]))
        return {"sample": item["sample"], "pieces": len(result["features"]), "files": [str(tissue)]}
    return _each_image(settings, report, work)


def artifact_batch(settings, folder, report):
    """Blood clots or Image quality for each ticked image: found with the chosen model (or, for Image quality without
    one, the detector settings) inside the image's Work within, and saved as its artifact polygons (the earlier ones
    stay in History). An image with saved polygons is kept as it is unless Replace is on. With Subtract on, each
    image's saved polygons are then cut out of its Work within annotation, as Subtract from tissue does."""
    from . import annotation, annotation_edits as edits, annotation_inputs as inputs, artifact_learning as learning
    from .settings import read_config

    root = Path(settings["root"])
    config = read_config(root)
    part = settings["part"]
    output = annotation.all_folders(root, config.get("directory_mappings", {}))["output"]
    values = learning.options(root, part)
    version = settings.get("version") or ""
    if settings.get("find") and not version and part == "clot":
        raise ValueError("Choose a Blood clots model first.")
    images = settings["images"]
    done, kept, failed = [], [], []
    for number, image in enumerate(images):
        name, sample = Path(image).name, annotation.sample_id(Path(image).name)
        step = lambda text, share: report(f"{name}: {text} ({number + 1} of {len(images)})", (number + share) / len(images))
        try:
            target = learning.polygons_path(output, sample, part)
            existing, revision = edits.read(target, kind="artifacts")
            line = {"image": name}
            if settings.get("find"):
                if existing and not settings.get("replace"):
                    kept.append(name)
                    line["kept"] = len(existing)
                else:
                    scope = inputs.scope_path(root, image, output, part)
                    tissue, _ = edits.read(scope, kind="tissue")
                    if not tissue:
                        raise ValueError("no saved Tissue or Work within annotation")
                    step("finding", 0)
                    if version:
                        found = learning.predict(root, part, version, image, values["scale"], tissue, values["cut"], values["minimum_um2"],
                                                 fill_holes_um2=float(values.get("fill_holes_um2", 0) or 0),
                                                 report=lambda text, share: step(text.lower(), .8 * share))
                    else:
                        found = learning.detect(image, config, part, values["scale"], values.get("value"), tissue, values["minimum_um2"],
                                                fill_holes_um2=float(values.get("fill_holes_um2", 0) or 0),
                                                scope_path=scope, detector_values=values.get("detector_values"), detectors=values.get("detectors"))
                    edits.replace(target, found["features"], revision, "Found for the ticked images", kind="artifacts")
                    line["found"] = len(found["features"])
            if settings.get("subtract"):
                # Each kind's saved polygons from the annotation chosen in that kind's Work within (both kinds: one after
                # the other, each from the result of the one before).
                line["subtracted"] = 0
                for kind_part in settings.get("parts") or [part]:
                    saved = learning.polygons_path(output, sample, kind_part)
                    polygons, revision = edits.read(saved, kind="artifacts")
                    if not polygons:
                        continue
                    step("subtracting " + learning.NAMES[kind_part], .9)
                    inputs.prepare_working(root, config, image, output, kind_part)
                    working = inputs.working_path(root, image, output, kind_part)
                    if not working.is_file():
                        raise ValueError("no saved Tissue or Work within annotation to subtract from")
                    kind = "tissue" if working.resolve() == annotation.valid_tissue(output, sample).resolve() else "regions"
                    edits.subtract(working, [saved], edits.revision(working), kind=kind,
                                   class_name=inputs.selection(root, image, kind_part).get("class_name", ""),
                                   label=annotation.tissue_label(config), artifact_revisions={str(saved): revision})
                    line["subtracted"] += len(polygons)
            done.append(line)
            step("done", 1)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            failed.append({"image": name, "error": str(error)})
            step(f"not done: {error}", 1)
    if not done and failed:
        raise ValueError(failed[0]["image"] + ": " + failed[0]["error"])
    return {"images": done, "kept": kept, "failed": failed, "count": len(done), "part": part}


def tls_candidates(settings, folder, report):
    from .tls import run
    return run(settings, folder, report)


def spatial_density(settings, folder, report):
    from . import spatial
    return spatial.density_job(settings, folder, report)


def spatial_triads(settings, folder, report):
    from . import spatial
    return spatial.triads_job(settings, folder, report)


def spatial_screening(settings, folder, report):
    from . import spatial
    return spatial.screening_job(settings, folder, report)


def spatial_umap(settings, folder, report):
    from . import spatial
    return spatial.umap_job(settings, folder, report)


def remove_small_pieces(settings, folder, report):
    """Delete the pieces of each listed image's Tissue (the annotation in its Work within) smaller than the area given,
    as Remove smaller does in the Tissue list; the earlier version stays in History."""
    from spacemap.viewer.shapes import area_px
    from spacemap.viewer.sources import image_info

    from . import annotation, annotation_edits as edits, annotation_inputs as inputs
    from .settings import read_config

    root = Path(settings["root"])
    config = read_config(root)
    output = annotation.all_folders(root, config.get("directory_mappings", {}))["output"]
    smallest = float(settings["min_um2"])
    done, failed = [], []
    for number, image in enumerate(settings["images"]):
        name, sample = Path(image).name, annotation.sample_id(Path(image).name)
        report(f"{name}: removing pieces under {smallest:g} µm² ({number + 1} of {len(settings['images'])})", number / len(settings["images"]))
        try:
            inputs.prepare_working(root, config, image, output, "tissue")
            target = inputs.working_path(root, image, output, "tissue")
            kind = "tissue" if target.resolve() == annotation.valid_tissue(output, sample).resolve() else "regions"
            label = annotation.tissue_label(config)
            features, revision = edits.read(target, kind=kind, label=label)
            if not features:
                raise ValueError("no saved Tissue")
            microns = image_info(str(image)).pixel_size_um or float(annotation.settings(config).get("microns_per_pixel", 0.5083))
            limit = smallest / microns ** 2
            only = inputs.selection(root, image, "tissue").get("class_name", "")
            keep = [f for f in features if (only and inputs.annotation_class(f) != only) or area_px(f) >= limit]
            if len(keep) < len(features):
                edits.replace(target, keep, revision, f"Removed pieces smaller than {smallest:g} µm²", kind=kind, label=label)
            done.append({"image": name, "removed": len(features) - len(keep), "kept": len(keep)})
        except (ValueError, OSError, KeyError) as error:
            failed.append({"image": name, "error": str(error)})
            report(f"{name}: not done: {error}", (number + 1) / len(settings["images"]))
    if not done and failed:
        raise ValueError(failed[0]["image"] + ": " + failed[0]["error"])
    return {"images": done, "failed": failed, "count": len(done), "min_um2": smallest}


def classify_shared(settings, folder, report):
    """Shared classes for the listed images: a class model fitted on a fair sample of each (unless one is given),
    chosen for the project, then every image classified with it; each finished image opens with the model's names."""
    from . import annotation, class_models
    from .settings import read_config

    root = Path(settings["root"])
    config = read_config(root)
    output = annotation.all_folders(root, config.get("directory_mappings", {}))["output"]
    images = settings["images"]
    skipped = []
    if settings.get("model"):
        model = settings["model"]
        record, _ = class_models.read(root, model)
        share = 0.0
    else:
        record = class_models.fit(root, config, images, output, settings["options"], report, settings.get("name", ""), settings.get("scale_by", "image"))
        model, skipped = record["id"], list(record["skipped"])
        class_models.select(root, model)
        share = 0.6
    done, failed = [], []
    for number, image in enumerate(images):
        name = Path(image).name
        report(f"{name}: classifying with {record['name']} ({number + 1} of {len(images)})", share + (1 - share) * number / len(images))
        try:
            found, seconds = class_models.classify(root, config, image, output, model)
            check = found["check"]
            done.append({"image": name, "far_percent": check["far_percent"], "differ": [item["id"] for item in class_models.flagged(check)],
                         "seconds": round(seconds, 1)})
        except (ValueError, OSError, RuntimeError) as error:
            failed.append({"image": name, "error": str(error)})
            report(f"{name}: not classified: {error}", share + (1 - share) * (number + 1) / len(images))
    if not done and failed:
        raise ValueError(failed[0]["image"] + ": " + failed[0]["error"])
    return {"model": model, "name": record["name"], "n_clusters": record["n_clusters"], "fitted": not settings.get("model"),
            "images": done, "failed": failed, "skipped": skipped, "count": len(done)}


def save_class_polygons(settings, folder, report):
    """Save polygons for each listed image from its last classification and its reviewed names (a class model's
    names for an image classified with one), as Save polygons does for the open image."""
    from . import annotation, annotation_edits, annotation_inputs, annotation_layers, class_models, classification
    from .settings import read_config

    root = Path(settings["root"])
    config = read_config(root)
    output = annotation.all_folders(root, config.get("directory_mappings", {}))["output"]
    done, failed = [], []
    for number, image in enumerate(settings["images"]):
        name = Path(image).name
        report(f"{name}: saving polygons ({number + 1} of {len(settings['images'])})", number / len(settings["images"]))
        try:
            last = classification.latest(root, image)
            loaded = classification.load(root, image, last["run_id"]) if last else None
            if not loaded:
                raise ValueError("not classified yet")
            prepared, found = loaded
            scope = found["provenance"]["scope"]
            if scope != str(annotation_inputs.scope_path(root, image, output, "classes")) or annotation_edits.revision(scope) != found["provenance"]["scope_revision"]:
                raise ValueError("its Work within changed since it was classified: classify it again")
            shared = found["provenance"].get("class_model")
            saved = classification.read_json(classification.directory(root, image) / (found["run_id"] + ".json"), {})
            state = classification.review(class_models.review_of(root, shared) if shared else saved.get("review"),
                                          np.flatnonzero(found["counts"]).tolist())
            target = annotation_layers.classification_target(output, image, annotation_inputs.selection(root, image, "classes"))
            polygons = classification.save_polygons(root, image, output, scope, prepared, found, state, annotation_edits.revision(target),
                                                    target=target, tidy=settings.get("tidy"))
            state["sealed"] = True
            classification.save_review(root, image, found, saved.get("options") or last["options"], state)
            done.append({"image": name, "polygons": polygons["count"]})
        except (ValueError, OSError, KeyError) as error:
            failed.append({"image": name, "error": str(error)})
            report(f"{name}: not saved: {error}", (number + 1) / len(settings["images"]))
    if not done and failed:
        raise ValueError(failed[0]["image"] + ": " + failed[0]["error"])
    return {"images": done, "failed": failed, "count": len(done)}


def subtract_annotations(settings, folder, report):
    """Subtract annotations on each ticked image by name (Stroma = tissue - Tumor, say), each a new editable layer."""
    from . import annotation, annotation_difference, annotation_workflow
    from .settings import read_config

    root = Path(settings["root"])
    config = read_config(root)
    output = annotation.all_folders(root, config.get("directory_mappings", {}))["output"]
    done, failed = [], []
    for number, image in enumerate(settings["images"]):
        name = Path(image).name
        report(f"{name}: subtracting ({number + 1} of {len(settings['images'])})", number / len(settings["images"]))
        try:
            made = annotation_difference.create_by_names(root, config, image, output, settings["label"], settings["base"], settings["subtract"])
            done.append({"image": name, "count": made["count"], "missing": made["missing"]})
            try:
                annotation_workflow.record(root, image=image, target=made["path"], event={"operation": "subtract_annotations", "parameters": made["recipe"]})
            except Exception as error:  # the layer is saved; only the index lags
                print(f"Workflow record failed: {error}", flush=True)
        except (ValueError, OSError, KeyError) as error:
            failed.append({"image": name, "error": str(error)})
            report(f"{name}: not done: {error}", (number + 1) / len(settings["images"]))
    if not done and failed:
        raise ValueError(failed[0]["image"] + ": " + failed[0]["error"])
    return {"images": done, "failed": failed, "count": len(done), "label": settings["label"]}


def train_artifact_model(settings, folder, report):
    from . import artifact_learning
    if settings.get('captures'):
        # Several reviewed images at once: each image's labels (frozen at click time) are sampled, replacing that
        # image's earlier examples, then one model is trained on them with the base examples of the other images.
        root, part, captures = settings['root'], settings['part'], settings['captures']
        fresh, skipped = [], list(settings.get('skipped', []))
        for number, item in enumerate(captures):
            name = Path(item['image']).name
            share = lambda message, progress: report(f"{name}: {message} ({number + 1} of {len(captures)})", .7 * (number + progress) / len(captures))
            try:
                made = artifact_learning.snapshot(root, part, item['image'], item['scale'], item['positives'], item['negatives'],
                                                  item['tissue'], item['remaining_good'], report=share)
            except ValueError as error:
                skipped.append(f"{name}: {error}")
                continue
            record = artifact_learning._read(artifact_learning.folder(root, part) / (made['snapshot_id'] + '.json'), {})
            record.pop('annotations', None)
            fresh.append(record)
        if not fresh:
            raise ValueError('No ticked image could be used: ' + '; '.join(skipped[:4]))
        keys = {item['image_key'] for item in fresh}
        request = {**settings, 'examples': [e for e in settings['examples'] if e['image_key'] not in keys] + fresh,
                   'current_snapshot': fresh[-1]['id']}
        request.pop('captures')
        result = artifact_learning.train(request, lambda message, progress: report(message, .7 + .3 * progress))
        return {**result, 'trained_on': [item['image'] for item in fresh], 'skipped': skipped}
    if settings.get('capture'):
        # Retain the base examples chosen at click time, even if later edits or
        # model selection changes occur while the worker is preparing features.
        root, part = settings['root'], settings['part']
        captured = artifact_learning.snapshot(root, part, **settings['capture'],
            report=lambda message, progress: report(message, .8*progress))
        identifier = captured['snapshot_id']
        current = artifact_learning._read(artifact_learning.folder(root, part)/(identifier+'.json'), {})
        current.pop('annotations', None)
        request = {**settings, 'current_snapshot': identifier,
            'examples': [e for e in settings['examples'] if e['image_key'] != current['image_key']] + [current]}
        request.pop('capture')
        result = artifact_learning.train(request, lambda message, progress: report(message, .8+.2*progress))
        result['preparation'] = current.get('execution', {})
        return result
    return artifact_learning.train(settings, report)


def _each_image(settings, report, work):
    """Run *work* for each listed image; with all images, one without its input is skipped."""
    done, skipped = [], []
    for index, item in enumerate(settings["images"]):
        report(f"{item['sample']}: starting", index / len(settings["images"]))
        try:
            done.append(work(item))
        except ValueError as error:
            if len(settings["images"]) == 1:
                raise
            skipped.append(f"{item['sample']}: {error}")
    return {"images": done, "skipped": skipped, "count": len(done)}


def apply_pieces(settings, folder, report):
    """Tab 4 step 2b for each listed image with the saved piece settings."""
    from . import annotation, pieces

    values = pieces.settings(settings["config"])
    from spacemap.viewer.sources import image_info
    return _each_image(settings, report, lambda item: pieces.apply(item["output"], item["sample"],
                       {**values, "microns_per_pixel":image_info(item["image"]).pixel_size_um or values["microns_per_pixel"]},
                       expected=item.get("tissue_revision"), label=annotation.tissue_label(settings["config"]), tissue_path=item.get("tissue_path"),
                       kind=item.get('tissue_kind','tissue'), class_name=item.get('class_name','')))


def make_zones(settings, folder, report):
    """Tab 4 step 4 for each listed image with the saved pool and zone rules."""
    from . import zones

    return _each_image(
        settings,
        report,
        lambda item: zones.apply(item["output"], item["zones"], item["sample"], settings["config"]),
    )


def export_classes(settings, folder, report):
    """Tab 4 step 3's class and domain files with the classes named in Studio."""
    from . import pixelclasses

    if not Path(settings["image"]).is_file():
        raise ValueError(f"The image {Path(settings['image']).name} is no longer there.")
    found = pixelclasses.export(
        settings["image"], settings["config"], settings["output"], settings["sample"], lambda text: report(text, None), scope=settings.get("scope")
    )
    return {"sample": settings["sample"], **found}
