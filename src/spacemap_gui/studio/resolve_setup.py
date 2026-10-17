"""Studio's editable RESOLVE workspace, using classic's gates and rule language.

Threshold calculations and reference-derived rules are drafts. Only Save writes
the classic panel/cache; only Run writes cell labels.
"""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from spacemap import dataio
from spacemap_workflows.leiden_job import write_json
from . import settings

from spacemap.thresholding_engine import THRESHOLD_METHODS, METHOD_ALIASES, parameters, catalog
MODES = ["Predefined (CSV)", *THRESHOLD_METHODS, "Z1 score (no threshold files)", "None (No Thresholds)"]
STATS = ["median", "mean", "min", "percentile_05", "percentile_25", "percentile_75"]
EDITABLE = {"threshold_percentile", "threshold_manual", "markers", "input_space", "threshold_mode", "threshold_source", "threshold_statistic",
            "tie_score_transform", "tie_score_cofactor", "z1_cutoff", "norm_z1", "norm_z2", "norm_cdf",
            "norm_nlog", "norm_order", "output_col", "unclassified_label", "enable_filter", "filter_col", "filter_val", "adaptive_hierarchy"}


def paths(root, table):
    cache = Path(root) / "outputs/phenotyping/cache"
    panel = cache / "resolve_pheno_panel.json"
    legacy = cache / f"resolve_pheno_{hashlib.md5(str(table).encode()).hexdigest()}.json"
    if not panel.is_file() and legacy.is_file():
        panel = legacy
    return {
        "panel": str(panel),
        "thresholds": str(cache / "resolve_thresholds_table.csv"),
        "settings": str(cache / "resolve_params.json"),
    }


def revision(root, table):
    from .tasks import source_revision
    value = {"source": source_revision({"table": str(table), **paths(root, table)}),
             "options": settings.resolve_values(root)}
    return settings.revision(value)


def rules_payload(text, markers):
    from spacemap_gui.resolve_rule_parser import parse_resolve_rule_text
    rows, errors = parse_resolve_rule_text(text, markers)
    if errors:
        raise ValueError("\n".join(errors[:12]))
    if not rows:
        raise ValueError("Add at least one rule, for example: CD8 = [CD3e, CD8].")
    criteria = [f"Criteria {i + 1}" for i in range(max(len(row["alternatives"]) for row in rows))]
    return {"criteria_cols": criteria, "table": [
        {"Phenotype": row["Phenotype"],
         **{col: row["alternatives"][i] if i < len(row["alternatives"]) else [] for i, col in enumerate(criteria)},
         "Supportive": row["support"], "Must not have": row["excluded"]} for row in rows]}


def rules_text(payload):
    lines = []
    for row in payload.get("table", []):
        groups = ["[" + ", ".join(row[col]) + "]" for col in payload.get("criteria_cols", []) if row.get(col)]
        if not row.get("Phenotype") or not groups:
            continue
        line = row["Phenotype"] + " = " + " OR ".join(groups)
        for col, directive in [("Supportive", "SUPPORT"), ("Must not have", "NOT")]:
            if row.get(col):
                line += "; " + directive + "=[" + ", ".join(row[col]) + "]"
        lines.append(line)
    return "\n".join(lines)


def get(root, table):
    from .phenotyping_inputs import column_roles
    from .background import folders
    from spacemap.viewer.reference import columns as references
    files = paths(root, table)
    options = settings.resolve_values(root)
    options["threshold_mode"] = METHOD_ALIASES.get(options.get("threshold_mode"), options.get("threshold_mode"))
    payload = json.loads(Path(files["panel"]).read_text()) if Path(files["panel"]).is_file() else {}
    thresholds = pd.read_csv(files["thresholds"]) if Path(files["thresholds"]).is_file() else pd.DataFrame()
    available = dataio.columns(table)
    # Old classic panels may predate the saved marker picker.
    used = list(dict.fromkeys(m for row in payload.get("table", [])
                             for key in [*payload.get("criteria_cols", []), "Supportive", "Must not have"]
                             for m in row.get(key, [])))
    options["markers"] = options.get("markers") or [m for m in used if m in available]
    options.setdefault("adaptive_hierarchy", True)
    config = settings.read_config(root)
    from .markers import suffix_of
    suffix = suffix_of(config, available)
    options.setdefault("input_space", "arcsinh" if options["markers"] and all(m.endswith(suffix) for m in options["markers"]) else "raw")
    thresholds = thresholds.replace([np.inf, -np.inf], np.nan)
    directory = folders(root, config.get("directory_mappings", {}), config)[1]
    return {"values": options, "revision": revision(root, table), "rules": rules_text(payload),
            "thresholds": thresholds.astype(object).where(pd.notna(thresholds), None).to_dict("records"),
            "references": references(table), "roles": column_roles(root, table),
            "threshold_folder": str(directory), "thresholding": catalog(), "modes": MODES, "columns": available}


def validate_options(options, available):
    import re
    markers = options.get("markers") or []
    if (not isinstance(markers, list) or not markers or len(markers) > 256
            or any(not isinstance(m, str) or m not in available for m in markers) or len(set(markers)) != len(markers)):
        raise ValueError("Choose markers with measurements in the combined table.")
    parameters({"percentile": options.get("threshold_percentile", 95), "manual": options.get("threshold_manual", 0)})
    if options.get("threshold_mode") not in [*MODES, *METHOD_ALIASES]:
        raise ValueError("Choose a listed threshold method.")
    if options.get("input_space", "raw") not in {"raw", "arcsinh"}:
        raise ValueError("Choose measured or already-arcsinh values.")
    if options.get("input_space") == "arcsinh" and options.get("tie_score_transform") != "identity_residual":
        raise ValueError("Already-arcsinh values need Scores = No extra transformation.")
    if not re.fullmatch(r"phenotype_[A-Za-z0-9_]{1,118}", options.get("output_col", "")):
        raise ValueError("Start the output column with phenotype_ and use letters, numbers and underscores.")
    if not isinstance(options.get("unclassified_label"), str) or not 0 < len(options["unclassified_label"].strip()) <= 128:
        raise ValueError("Enter a short name for unmatched cells.")
    if options.get("enable_filter") and (options.get("filter_col") not in available or str(options.get("filter_val") or "").strip() == ""):
        raise ValueError("Choose a filter column and a value, or switch the filter off.")
    if options.get("threshold_source", "Auto") not in {"Auto", "Polygon values", "Final scalar"}:
        raise ValueError("Choose Auto, Polygon values or Final scalar.")
    if options.get("threshold_statistic", "median") not in STATS:
        raise ValueError("Choose a listed background statistic.")
    for key in ("norm_z1", "norm_z2", "norm_cdf", "norm_nlog", "adaptive_hierarchy"):
        if key in options and not isinstance(options[key], bool):
            raise ValueError("Choose on or off for normalization.")
    order = options.get("norm_order", [])
    if (not isinstance(order, list) or len(order) != len(set(order))
            or set(order) - {"Z1", "Z2", "CDF", "NLog"}):
        raise ValueError("Choose each normalization step only once.")
    enabled = {step for step, key in [("Z1", "norm_z1"), ("Z2", "norm_z2"), ("CDF", "norm_cdf"), ("NLog", "norm_nlog")] if options.get(key)}
    if set(order) != enabled:
        raise ValueError("The normalization steps and enabled options disagree. Save the visible settings again.")
    if options.get("norm_nlog") and (not options.get("norm_cdf") or "NLog" not in order or "CDF" not in order or order.index("NLog") < order.index("CDF")):
        raise ValueError("Put CDF before −log(1 − CDF).")
    if options.get("tie_score_transform") not in {"identity_residual", "asinh_difference", "log1p_difference"}:
        raise ValueError("Choose a listed score transformation.")
    for key, low, high in [("tie_score_cofactor", 0.0001, 65535), ("z1_cutoff", -10, 20)]:
        if not np.isfinite(float(options.get(key, 5 if key == "tie_score_cofactor" else 1.5))) or not low <= float(options.get(key, 5 if key == "tie_score_cofactor" else 1.5)) <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    return markers


def filtered(frame, options):
    from spacemap_gui.tabs.tab7_phenotyping.common import apply_data_filter
    if options.get("enable_filter"):
        return apply_data_filter(frame, options)
    return frame


def _rois(root, table):
    from .phenotyping_inputs import column_roles
    roi = column_roles(root, table)["roi"]
    if not roi:
        raise ValueError("Choose the ROI column at the top of Phenotyping.")
    names = dataio.load(table, columns=[roi])[roi]
    if names.isna().any() or names.astype(str).str.strip().eq("").any():
        raise ValueError("Every cell needs an ROI name.")
    return roi, names.astype(str)


def threshold_frame(rows, markers, rois, *, complete=True):
    """Match by canonical ROI, rejecting ambiguous or incomplete gates."""
    from spacemap.viewer.catalog import sample_of
    frame = pd.DataFrame(rows)
    if frame.empty:
        if complete:
            raise ValueError("Import saved thresholds or calculate them here first.")
        return frame
    if "batch_id" not in frame or not set(markers) <= set(frame):
        raise ValueError("The threshold table needs an ROI and every selected marker.")
    names = frame["batch_id"].astype(str).map(lambda name: sample_of(name).casefold())
    if names.duplicated().any():
        raise ValueError("Several threshold rows match the same ROI. Keep one row per ROI.")
    lookup = dict(zip(names, frame.index))
    output = []
    for roi in pd.unique(rois):
        key = sample_of(roi).casefold()
        if key not in lookup:
            if complete:
                raise ValueError(f"No thresholds for {roi}. Load or calculate thresholds for every ROI.")
            continue
        row = frame.loc[lookup[key]].to_dict()
        row["batch_id"] = str(roi)
        output.append(row)
    result = pd.DataFrame(output)
    if not result.empty:
        result[markers] = result[markers].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(result[markers].to_numpy(float)).all():
            raise ValueError("Every selected marker needs a finite threshold in each ROI.")
    return result


def calculate(root, table, options, *, background_folder=None):
    """Calculate an editable draft; never modify Background or cell labels."""
    from spacemap_gui.tabs.tab7_phenotyping.common import compute_auto_thresholds, _THRESHOLD_MODE_TO_KEY, _load_thresholds_table
    markers = validate_options(options, dataio.columns(table))
    roi, names = _rois(root, table)
    if background_folder is not None:
        from .markers import suffix_of
        suffix = suffix_of(settings.read_config(root), dataio.columns(table))
        if options.get("input_space") == "arcsinh" or any(m.endswith(suffix) for m in markers):
            raise ValueError("Background thresholds use raw values. Choose raw markers, or calculate local thresholds for corrected values.")
        directory = Path(background_folder).expanduser()
        if not directory.is_dir():
            raise ValueError("Choose the folder with Background threshold CSV files.")
        draft = _load_thresholds_table(str(directory), markers, names.unique().tolist(),
                                       options.get("threshold_marker_col") or None, options.get("threshold_value_col") or None,
                                       options.get("polygon_prefix", "Polygon"), options.get("threshold_statistic", "median"),
                                       options.get("threshold_source", "Auto"))
        draft = threshold_frame(draft.to_dict("records"), markers, names)
    else:
        method = _THRESHOLD_MODE_TO_KEY.get(options["threshold_mode"])
        if method in (None, "predefined"):
            raise ValueError("Choose a threshold method to calculate gates.")
        # Read measurements once (Parquet column projection), not one entire table per ROI.
        frame = dataio.load(table, columns=list(dict.fromkeys([roi, *markers, *([options["filter_col"]] if options.get("enable_filter") else [])])))
        frame[roi] = frame[roi].astype(str)
        frame = filtered(frame, options)
        if frame.empty:
            raise ValueError("No cells match the filter.")
        good = np.isfinite(frame[markers].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
        draft = compute_auto_thresholds(frame.loc[good], markers, method, batch_col=roi, params={"percentile": options.get("threshold_percentile", 95), "manual": options.get("threshold_manual", 0)})
        draft = threshold_frame(draft.to_dict("records"), markers, frame[roi])
    return {"thresholds": draft.astype(object).where(pd.notna(draft), None).to_dict("records")}


def save(root, table, body):
    from spacemap_gui import persistence
    # Ignore non-editable values echoed by the UI; never import arbitrary paths
    # or old classic filter settings from a request.
    options = {**settings.resolve_values(root), **{k: v for k, v in body.get("values", {}).items() if k in EDITABLE}}
    markers = validate_options(options, dataio.columns(table))
    payload = rules_payload(body.get("rules", ""), markers)
    roi, rois = _rois(root, table)
    if options.get("enable_filter"):
        scope = dataio.load(table, columns=list(dict.fromkeys([roi, options["filter_col"]])))
        rois = filtered(scope, options)[roi].astype(str)
    rows = body.get("thresholds", [])
    threshold = threshold_frame(rows, markers, rois) if options["threshold_mode"] == "Predefined (CSV)" else pd.DataFrame()
    options["batch_col"] = roi
    options["transformation"] = {"identity_residual": "None", "asinh_difference": "ArcSinh", "log1p_difference": "Log"}[options["tie_score_transform"]]
    options["normalization_slots"] = (options["norm_order"] + ["None"] * 4)[:4]
    # Classic must see the exact same gate and rule selection after reopening.
    cache = Path(root) / "outputs/phenotyping/cache"
    files = paths(root, table)
    with settings.project_lock(root):
        if body.get("revision") != revision(root, table):
            raise ValueError("RESOLVE settings or cell data changed. Reload the panel before saving.")
        cache.mkdir(parents=True, exist_ok=True)
        targets = [cache / "resolve_pheno_panel.json", Path(files["thresholds"]), cache / "resolve_params.json", settings.config_path(root)]
        previous = {path: path.read_bytes() if path.exists() else None for path in targets}
        try:
            write_json(cache / "resolve_pheno_panel.json", payload)
            if not threshold.empty:
                temporary = cache / ".resolve_thresholds_table.csv.tmp"
                threshold.to_csv(temporary, index=False)
                temporary.replace(files["thresholds"])
            write_json(cache / "resolve_params.json", options)
            config = settings.read_config(root)
            persistence.update_stored_config_key(config, "resolve_config", options)
            persistence.save_project_config(str(root), config)
            if settings.read_config(root).get("tab_configurations", {}).get("tab7_phenotyping", {}).get("resolve_config") != options:
                raise ValueError("The project settings did not save. Check folder permissions.")
        except Exception:
            for path, data in previous.items():
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(data)
            raise
        from spacemap.viewer.live_resolve import invalidate_settings
        invalidate_settings(cache / "resolve_params.json")
    return get(root, table)


def draft_reference(root, table, body):
    """Conservative AND rules from reference cells' actual binary gate patterns.

    Show training coverage/purity, never claim held-out accuracy. The user reviews
    the text before saving or applying it. Avoid huge scans of unlabelled markers.
    """
    from spacemap.viewer.reference import columns
    from .tasks import resolve_scores
    options = {**settings.resolve_values(root), **body.get("values", {})}
    markers = validate_options(options, dataio.columns(table))
    reference = body.get("reference")
    if reference not in columns(table):
        raise ValueError("Choose a Manual reference column.")
    roi, names = _rois(root, table)
    frame = dataio.load(table, columns=list(dict.fromkeys([roi, reference, *markers, *([options["filter_col"]] if options.get("enable_filter") else [])])))
    frame[roi] = names
    frame = filtered(frame, options)
    frame = frame.loc[np.isfinite(frame[markers].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)]
    if frame.empty:
        raise ValueError("No cells with complete measurements match the filter.")
    thresholds = pd.DataFrame(body.get("thresholds", []))
    binary, _ = resolve_scores(frame, thresholds, markers, options, roi)
    labels = frame[reference]
    use = labels.notna() & labels.astype(str).str.strip().ne("") & ~labels.astype(str).str.lower().isin(["others", "unassigned"])
    use &= np.isfinite(frame[markers].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    x, y = binary.loc[use].to_numpy(bool), labels[use].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Label at least five cells in each of two phenotypes before drafting rules.")
    lines, evidence, skipped = [], [], []
    for name in pd.unique(y):
        target = y == name
        if target.sum() < 5:
            skipped.append(name + ": fewer than 5 reference cells")
            continue
        selected, matches = [], np.ones(len(y), dtype=bool)
        # At most 3 required markers; each addition must improve F1 and retain
        # at least half this label's reference cells. Negatives are NOT hard gates
        # in classic RESOLVE, so don't invent exclusion rules from binary absences.
        best_score = 0.0
        for _ in range(min(3, len(markers))):
            candidates = []
            for j in range(len(markers)):
                if j in selected:
                    continue
                matched = matches & x[:, j]
                tp = int((matched & target).sum())
                recall = tp / int(target.sum())
                precision = tp / max(1, int(matched.sum()))
                score = 2 * precision * recall / max(1e-12, precision + recall)
                if recall >= .5 and tp >= 5:
                    candidates.append((score, precision, j, matched))
            if not candidates:
                break
            score, precision, j, matched = max(candidates, key=lambda c: (c[0], c[1], -c[2]))
            if score <= best_score + .001:
                break
            selected.append(j); matches = matched; best_score = score
        tp = int((matches & target).sum())
        precision = tp / max(1, int(matches.sum()))
        if not selected or precision < .8:
            skipped.append(name + ": no clear positive-marker rule")
            continue
        lines.append(name + " = [" + ", ".join(markers[j] for j in selected) + "]")
        evidence.append({"phenotype": name, "labelled": int(target.sum()), "matched": tp,
                         "purity": precision, "coverage": tp / int(target.sum())})
    return {"rules": "\n".join(lines), "evidence": evidence, "skipped": skipped,
            "note": "Draft from your labelled cells only. Coverage and purity describe these examples, not independent accuracy. Review before saving."}


def job_settings(root, table, args):
    from .phenotyping_inputs import column_roles, segmentation_column
    from .tasks import source_revision, validate_table
    saved = get(root, table)
    options = saved["values"]
    if args.get("settings_revision") and args["settings_revision"] != saved["revision"]:
        raise ValueError("RESOLVE changed after saving. Save the visible settings again.")
    markers = validate_options(options, dataio.columns(table))
    rules_payload(saved["rules"], markers)
    roles = column_roles(root, table)
    result = {**paths(root, table), "table": table, "markers": markers, "options": options,
              "output_col": args["column"], "overwrite_columns": [args["column"]] if args.get("overwrite") else [],
              "cell_id_col": roles["cell_id"], "roi_column": roles["roi"],
              "label_column": segmentation_column(root, table)}
    validate_table(result)
    if options["threshold_mode"] == "Predefined (CSV)":
        rois = _rois(root, table)[1]
        if options.get("enable_filter"):
            frame = dataio.load(table, columns=list(dict.fromkeys([roles["roi"], options["filter_col"]])))
            rois = filtered(frame, options)[roles["roi"]].astype(str)
        threshold_frame(saved["thresholds"], markers, rois)
    result["revision"] = source_revision(result)
    return result
