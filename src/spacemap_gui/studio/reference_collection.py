"""One model from several per-image reference tables; predictions stay per table."""

from pathlib import Path

import numpy as np
import pandas as pd

from spacemap import dataio
from spacemap.viewer import consensus_model, reference
from spacemap.viewer.sources import fingerprint
from . import phenotyping_inputs as inputs, reference_models as models, tasks


def plan(root, table):
    found = inputs.plan(root, {}, table)
    tables = [{'path':row['table_path'], 'image':row['roi'], 'name':Path(row['table_path']).name}
              for row in found['rows'] if row.get('table_path')]
    return {'tables':tables, 'selected_images':found['selected_images'], 'per_image':found['per_image']}


def _stamp(source):
    reviews = reference.path_for(source["table"])
    return {"revision": tasks.source_revision(source), "marker_revision": models.marker_revision(source),
            "review_revision": fingerprint(reviews) if reviews.exists() else None}


def check_sources(sources):
    for source in sources:
        if any(source[key] != value for key, value in _stamp(source).items()):
            raise ValueError("Labels or measurements changed during training. Run training again.")


def prepare(channel, body):
    from spacemap_gui.viewer import registry
    from . import reference_examples, reference_features, reference_setup, mr_ml
    project = registry.session_project(channel) or {}
    root, active = project["root"], inputs.combined_table(channel)
    action = body.get("action")
    if action not in {"train", "predict"}:
        raise ValueError("Choose Train selected or Predict selected.")
    from . import phenotyping_images
    listed = plan(root, active)
    if 'images' in body or 'tables' not in body:
        selected_sources = phenotyping_images.sources(root, active, body.get('images'))
    else:
        requested = body['tables']
        allowed = {item['path'] for item in listed['tables']}
        if not isinstance(requested, list) or not requested or any(path not in allowed for path in requested):
            raise ValueError('Select tables listed in the current Tables folder.')
        selected_sources = phenotyping_images.sources(root, active,
            [item['image'] for item in listed['tables'] if item['path'] in requested])
    column = body.get("reference_column", "phenotype_ref1")
    if not isinstance(column, str) or not column.strip() or len(column) > 128:
        raise ValueError("Choose the reference column to learn from.")
    setup = reference_setup.training(root)
    model_id = body.get("model_id", setup["base_model_id"])
    if model_id:
        models.metadata(model_id)  # Validate before a job starts.
    if action == 'train':
        reference_examples.require_reusable({**setup, 'base_model_id': model_id})
    if action == "predict" and not model_id:
        raise ValueError("Choose a saved model to predict with.")
    features = reference_features.options(channel) if action == "train" else None
    markers = features["selected"] if features else models.metadata(model_id)["markers"]
    if not markers:
        raise ValueError("Choose measurements under Model training → Features.")
    sources, prediction_settings = [], []
    for selected_source in selected_sources:
        path = selected_source['table']
        available = dataio.columns(path)
        roles = inputs.column_roles(root, path, available)
        if not roles["cell_id"] or not inputs.segmentation_column(root, path):
            raise ValueError(f"{Path(path).name}: choose Cell ID and Segmentation label in Column roles.")
        if action == "train":
            if set(markers) - set(available):
                raise ValueError(f"{Path(path).name} is missing training features: " + ", ".join(sorted(set(markers) - set(available))))
            if column in markers or column in roles.values():
                raise ValueError("Reference labels and cell IDs cannot be training features.")
        source = {**selected_source, "table": path, "markers": markers, "roi_column": roles["roi"], "label_column": inputs.segmentation_column(root, path),
                  "cell_id_col": roles["cell_id"], "training_column": column, "checked_only": bool(body.get("checked_only"))}
        source['excluded_training_phenotypes'] = setup['excluded_training_phenotypes']
        source.update(_stamp(source))
        sources.append(source)
        if action == "predict":
            output = body.get("output_column", "phenotype_mr_ml")
            chosen = {**mr_ml.DEFAULTS, "workflow": "saved_model", "model_id": model_id,
                      "ground_truth_col": column, "enable_filter": False, "match_scale": False,
                      "output_phenotype_col": output, "confidence_col": output + "_confidence",
                      "refined_output_phenotype_col": output + "_refined"}
            if body.get('use_mrml_settings'):
                chosen = {**mr_ml.values(root), 'workflow': 'saved_model', 'model_id': model_id}
            prediction_settings.append({**mr_ml.job_settings(root, path, roles, chosen, overwrite=bool(body.get("overwrite"))),
                                        **selected_source})
    return {"action": action, "table": active, "project_root": root, "sources": sources, "prediction_settings": prediction_settings,
            **setup, "base_model_id": model_id, "markers": markers, "training_column": column,
            "feature_kinds": {item["name"]: item["group"] for item in features["features"] if item["name"] in markers} if features else {},
            "allow_missing": features["allow_missing"] if features else True}


def _examples(source, report):
    """Bounded per-class sample, even when one image contains only one phenotype."""
    if source["training_column"] not in dataio.columns(source["table"]):
        return np.empty(0, np.int64), np.empty(0, object)
    excluded, checked = set(), None
    with reference.database(source["table"]) as db:
        flags = list(db.execute("SELECT roi,label FROM reviews WHERE column_name=? AND flagged=1", (source["training_column"],))) if db else []
    from spacemap.viewer import cells
    for roi in {item[0] for item in flags}:
        _, positions = cells.assignment_rows(source["table"], roi, [item[1] for item in flags if item[0] == roi],
                                             roi_column=source["roi_column"], label_column=source["label_column"])
        excluded.update(positions.tolist())
    if source["checked_only"]:
        checked = set(models.checked_rows(source))
    rng = np.random.default_rng(42)
    reservoirs = {}
    for offset, values in consensus_model._label_batches(source):
        text = pd.Series(values).fillna("").astype(str).str.strip()
        keep = ~text.str.lower().isin(consensus_model._EMPTY).to_numpy()
        from spacemap.phenotype_policy import allowed
        keep &= allowed(text, source.get('excluded_training_phenotypes', []))
        positions = np.arange(offset, offset + len(values))
        from . import phenotyping_images
        keep &= phenotyping_images.rows_mask(source, positions)
        if excluded:
            keep &= ~np.isin(positions, list(excluded))
        if checked is not None:
            keep &= np.isin(positions, list(checked))
        for name in pd.unique(text[keep]):
            rows = positions[keep & text.eq(name).to_numpy()]
            old_rows, old_rank = reservoirs.get(name, (np.empty(0, np.int64), np.empty(0)))
            ranks = np.concatenate([old_rank, rng.random(len(rows))])
            rows = np.concatenate([old_rows, rows])
            selected = np.argsort(ranks)[-consensus_model.PER_CLASS:]
            reservoirs[name] = rows[selected], ranks[selected]
        if len(reservoirs) > 1000:
            raise ValueError("Choose a phenotype column with category names.")
        report(f"Reading labels: {offset + len(values):,} cells checked")
    rows = np.concatenate([value[0] for value in reservoirs.values()]) if reservoirs else np.empty(0, np.int64)
    targets = np.concatenate([np.repeat(name, len(value[0])) for name, value in reservoirs.items()]) if reservoirs else np.empty(0, object)
    order = np.argsort(rows)
    return rows[order], targets[order]


def run(settings, folder, report):
    from . import mr_ml
    from spacemap_workflows.leiden_job import write_json
    if settings["action"] == "predict":
        results = []
        jobs = settings["prediction_settings"]
        # Preflight every source before writing any prediction columns.
        for job in jobs:
            tasks.validate_mr_ml(job)
            if tasks.source_revision(job) != job["revision"]:
                raise ValueError("An input table changed. Start prediction again.")
        for index, job in enumerate(jobs):
            def progress(message, fraction=0):
                report(f"{Path(job['table']).name} ({index + 1}/{len(jobs)}): {message}", (index + fraction) / len(jobs))
            if hasattr(report, "protect_output"):
                progress.protect_output = report.protect_output
            results.append(mr_ml.predict_saved(job, Path(folder) / str(index), progress))
            write_json(Path(folder) / "completed_tables.json", results)
            if index + 1 < len(jobs) and hasattr(report, "finish_output"):
                # Every published table has a receipt before cancellation is allowed again.
                report.finish_output()
        return {"tables": results, "images": len(results), "cells": sum(item["cells"] for item in results),
                "column": jobs[0]["output_col"], "model_id": settings["base_model_id"]}
    check_sources(settings["sources"])
    chunks, origins, origin_rows = [], [], []
    for index, source in enumerate(settings["sources"]):
        def progress(message):
            report(f"{Path(source['table']).name}: {message}", .2 * index / len(settings["sources"]))
        rows, targets = _examples(source, progress)
        if not len(rows):
            continue
        values = consensus_model.selected_values(source, rows, streaming=True)
        frame = pd.DataFrame(values, columns=settings["markers"])
        frame[settings["training_column"]] = targets
        chunks.append(frame)
        origins.extend([source["table"]] * len(rows))
        origin_rows.extend(rows.tolist())
    if not chunks:
        raise ValueError("Add reference labels to the selected image tables first.")
    pooled = pd.concat(chunks, ignore_index=True)
    pooled["label"] = np.arange(1, len(pooled) + 1)
    path = Path(folder) / "reference_examples.parquet"
    pooled.to_parquet(path, index=False)
    training = {**settings, "table": str(path), "roi_column": "", "label_column": "label",
                "training_sources": settings["sources"], "training_origin": {"tables": origins, "rows": origin_rows}}
    training["revision"] = tasks.source_revision(training)
    return models.train(training, folder, lambda message, fraction=0: report(message, .2 + .8 * fraction))
