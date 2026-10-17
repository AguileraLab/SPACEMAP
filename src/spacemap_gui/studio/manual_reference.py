"""Manual labels on the combined table, independent of the currently open ROI."""

import json

import numpy as np
import pandas as pd

from spacemap import dataio, labels, pending_columns
from spacemap.viewer import cells, reference
from spacemap.viewer.sources import fingerprint
from . import data_view, phenotyping_inputs


def context(channel):
    from spacemap_gui.viewer import registry
    project = registry.session_project(channel) or {}
    path = phenotyping_inputs.combined_table(channel)
    roles = phenotyping_inputs.column_roles(project["root"], path)
    return path, {"roi_column":roles["roi"], "label_column":phenotyping_inputs.segmentation_column(project["root"], path)}, roles


def options(channel, column=""):
    from spacemap import phenotype_presets
    path, keys, roles = context(channel)
    table = data_view._table(path)
    refs = reference.columns(path)
    chosen = column or (refs[0] if refs else "phenotype_ref1")
    counts = data_view.summary(path, column=chosen, limit=200) if chosen in table.names else {}
    names = list(dict.fromkeys([item['name'] for item in phenotype_presets.read()['entries']] + sorted(set(reference.phenotype_names(path, chosen)) | set(reference.phenotype_names(path)))))
    return {"table":str(path), "revision":fingerprint(path), "rows":table.count, "references":refs, "column":chosen,
            "phenotypes":names, "counts":counts,
            "preset_colours":phenotype_presets.ensure(names),
            "columns":[name for name in table.names if table.kinds[name] == "group" and name not in roles.values() and name not in keys.values()],
            "writing":pending_columns.commit_status(path), "pending_labels":labels.pending_count(path),
            "pending_columns":pending_columns.pending_names(path)}


def create(channel, column):
    path, keys, roles = context(channel)
    with labels.write_lock(path):
        reference.validate_column(path, column, **keys)
        if column not in dataio.columns(path):
            table = data_view._table(path)
            anchor = roles["cell_id"]
            if anchor not in table.names:
                raise ValueError("Choose the Cell ID column above before creating a reference column.")
            ids = pd.Series(cells.values_at_rows(path, np.arange(table.count), anchor))
            if ids.isna().any() or ids.duplicated().any():
                raise ValueError("The Cell ID column must identify each cell once, with no missing IDs.")
            pending_columns.save(path, {column:pd.Series([None]*table.count, dtype=object)},
                                 anchor_column=anchor, anchor_values=ids, source_note="Manual reference")
        reference.register_column(path, column)
    return options(channel, column)


def _positions(channel, path, keys, selection):
    table = data_view._table(path)
    if not isinstance(selection, dict) or len(set(selection) & {"rows", "cells", "filters", "token"}) != 1:
        raise ValueError("Choose cells in the viewer, Data rows, filtered Data rows, or a marker rule.")
    if "token" in selection:
        from . import selection as rules
        item = rules.get(selection["token"], channel=channel, table=path)
        return item["groups"]
    if "cells" in selection:
        ids = selection["cells"]
        if not isinstance(ids, list) or not ids or len(ids) > 200000 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in ids):
            raise ValueError("Select 1–200,000 cells.")
        if selection.get("roi") is None:
            raise ValueError("The selected cells need an ROI.")
        return [{"roi":str(selection["roi"]), "ids":list(dict.fromkeys(ids))}]
    if "rows" in selection:
        rows = selection["rows"]
        if not isinstance(rows, list) or len(rows) > 200000 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < table.count for v in rows):
            raise ValueError("Select valid rows of the current combined table.")
        rows = np.array(list(dict.fromkeys(rows)), dtype=np.int64)
    else:
        filters = selection["filters"]
        if not isinstance(filters, list):
            raise ValueError("Choose Data filters first.")
        rows = data_view._view(table, filters, None)
    if not len(rows):
        raise ValueError("No cells are selected.")
    if len(rows) > 200000:
        raise ValueError("Narrow the selection to at most 200,000 cells for a manual assignment.")
    for name in keys.values():
        if name not in table.names:
            raise ValueError("Choose ROI and segmentation cell columns before assigning labels.")
    values = table.take(rows, list(keys.values()))
    frame = pd.DataFrame({"roi":values[keys["roi_column"]], "label":values[keys["label_column"]]})
    numbers = pd.to_numeric(frame.label, errors="coerce")
    if frame.roi.isna().any() or not np.isfinite(numbers).all() or (numbers < 1).any() or (numbers != np.floor(numbers)).any():
        raise ValueError("Some selected rows have missing or invalid ROI/cell IDs. Nothing was assigned.")
    frame["label"] = numbers.astype(np.int64)
    frame["roi"] = frame.roi.astype(str)
    if frame.duplicated(["roi", "label"]).any():
        raise ValueError("Some selected rows share the same ROI/cell ID. Nothing was assigned.")
    return [{"roi":roi, "ids":group.label.tolist()} for roi, group in frame.groupby("roi", sort=False)]


def assign(channel, body):
    path, keys, _ = context(channel)
    column, value = body.get("column"), body.get("phenotype")
    with labels.write_lock(path):
        if body.get("table") != str(path) or body.get("revision") != fingerprint(path):
            raise ValueError("The cell table changed. Refresh the selection before assigning labels.")
        reference.validate_column(path, column, **keys)
        groups = _positions(channel, path, keys, body.get("selection"))
        found = reference.save_groups(path, groups, column, value, body.get("request_id"), **keys)
    return {"assigned":found["assigned"], "request_id":body["request_id"], "column":column,
            "message":f"{found['assigned']:,} cells {'labelled ' + value if value else 'cleared'} in {column}."}


def undo(channel, request_id):
    path, keys, _ = context(channel)
    with reference.database(path) as db:
        row = db.execute("SELECT * FROM requests WHERE id=? AND done=1", (request_id,)).fetchone() if db else None
    if not row or row["fingerprint"] != fingerprint(path):
        raise ValueError("That edit cannot be undone after the table changed.")
    updates, result = json.loads(row["updates"]), json.loads(row["result"])
    if not updates or not isinstance(result.get("previous"), list):
        raise ValueError("Choose a manual selection edit to undo.")
    return reference.undo_groups(path, updates[0]["column"], result["previous"], updates[0]["value"], request_id + ":undo", **keys)


def evaluate(path, reference_column, method):
    """Score named predictions on labelled reference cells, including missing predictions."""
    if reference_column == method:
        raise ValueError("Choose a different prediction column to compare with your reference.")
    if reference_column not in reference.columns(path) or method not in dataio.columns(path):
        raise ValueError("Choose a manual reference and an existing prediction column.")
    frame = dataio.load(path, columns=[reference_column, method])
    from spacemap.viewer.suggestions import named
    truth = pd.Series(named(frame[reference_column]))
    predicted = pd.Series(named(frame[method]))
    keep = truth != ""
    truth, predicted = truth[keep], predicted[keep]
    if truth.empty:
        raise ValueError("Label some reference cells before checking agreement.")
    if predicted[predicted != ""].str.fullmatch(r"Cluster\s+\d+").all() and (predicted != "").any():
        raise ValueError("Name the clusters first, then choose their phenotype column. Cluster numbers are not phenotype names.")
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    names = sorted(set(truth) | (set(predicted) - {""}))
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=names, zero_division=0)
    return {"reference":reference_column, "method":method, "cells":len(truth), "unlabelled":int((predicted == "").sum()),
            "accuracy":float(accuracy_score(truth, predicted)), "macro_f1":float(f1.mean()),
            "phenotypes":[{"phenotype":name,"cells":int(n),"precision":float(p),"recall":float(r),"f1":float(f)}
                          for name,n,p,r,f in zip(names,support,precision,recall,f1)],
            "note":"Only manually labelled cells are scored. Missing predictions count as errors. Labels must use the same names. If these cells trained the model or helped name clusters, this is agreement, not an independent test."}


def training_settings(channel, column, checked_only=False):
    from . import reference_features, tasks
    path, keys, roles = context(channel)
    table = data_view._table(path)
    if column not in table.names or table.kinds[column] != "group" or column in keys.values():
        raise ValueError("Choose a phenotype column with category names, not a measurement or cell ID.")
    if column in roles.values():
        raise ValueError("Choose a phenotype column, not an ROI, slide or cell ID.")
    features = reference_features.options(channel)
    if features["missing"]:
        raise ValueError("Training features no longer in this table: " + ", ".join(features["missing"]) + ". Update Model training → Features.")
    if not features["selected"]:
        raise ValueError("Choose measurements in Model training → Features first.")
    kinds = {item["name"]:item["group"] for item in features["features"]}
    settings = {"table":path,"markers":features["selected"],"training_column":column,"checked_only":checked_only,
                "feature_kinds":{name:kinds[name] for name in features["selected"]},
                "allow_missing":features["allow_missing"],**keys}
    settings["revision"] = tasks.source_revision(settings)
    return settings
