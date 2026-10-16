"""Studio's Sub-phenotyping (Tab 7): chosen phenotypes split into functional states, with classic's steps.

Settings are classic's ``outputs/phenotyping/cache/subphenotyping_params.json`` (read and merged); the
steps are classic's (``spacemap.ph_subpheno_utils``), for each chosen phenotype on its own:

1. **Find markers** (a job, ``mode="discover"``): score every panel marker inside the phenotype
   against dataset-wide reference numbers (``score_markers``) on up to 40,000 of its cells, and
   call keep or drop with the reason (``recommend_markers``: flat here, batch-driven, possible spillover).
2. **Run** (a job, ``mode="run"``): the markers kept for that phenotype (the suggestion, or the user's
   choice), z-scored within batch (``build_feature_matrix``), clustered with Leiden, SOM or VORTEX and
   merged to N (``subcluster``), then checked (``validate_subclusters``: size, reproducible across
   batches, a separating marker; optional stability across re-runs).

The run saves one new column ``<output>_clusters`` ("<Phenotype>_<n>" for the chosen phenotypes'
cells) beside the table, with each group's size, top markers, check, a heatmap and where it is, so the
cluster explorer shows and names them into ``<output>``.
"""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

DISCOVERY_SAMPLE = 40000
DEFAULTS = {
    "parent_col": "phenotype_ml_consensus",
    "batch_col": "batch_id",
    "parent_values": [],
    "markers": [],
    "chosen_markers": {},
    "method": "Leiden",
    "resolution": 1.0,
    "n_neighbors": 30,
    "n_pcs": 20,
    "grid": 10,
    "iterations": 20000,
    "k_neighbors": 35,
    "min_cluster_size": 50,
    "n_meta_clusters": 10,
    "use_gpu": True,
    "subsample_size": 150000,
    "seed": 42,
    "check_stability": False,
    "collapsed_below": 0.40,
    "max_batch_eta2": 0.25,
    "max_markers": 0,
    # Classic's own output names (sub_phenotype) are not phenotype_ columns; Studio's are.
    "studio_output_col": "phenotype_sub",
}
EDITABLE = {
    "parent_col": "column",
    "parent_values": "list",
    "markers": "list",
    "chosen_markers": "mapping",
    "method": ("Leiden", "SOM", "Vortex"),
    "resolution": (float, 0.1, 3.0),
    "n_neighbors": (int, 5, 100),
    "grid": (int, 4, 30),
    "k_neighbors": (int, 5, 200),
    "n_meta_clusters": (int, 0, 60),
    "use_gpu": bool,
    "check_stability": bool,
    "collapsed_below": (float, 0.1, 0.9),
    "max_batch_eta2": (float, 0.05, 0.9),
    "studio_output_col": "output",
}
OUTPUT = re.compile(r"phenotype_[A-Za-z0-9_]{1,100}")


def cache_path(root):
    return Path(root) / "outputs" / "phenotyping" / "cache" / "subphenotyping_params.json"


def values(root):
    path = cache_path(root)
    saved = json.loads(path.read_text()) if path.is_file() else {}
    saved = saved if isinstance(saved, dict) else {}
    return {**DEFAULTS, **{key: value for key, value in saved.items() if value is not None and key in DEFAULTS}}


def check(changes):
    from .leiden import COLUMN

    if not isinstance(changes, dict) or set(changes) - set(EDITABLE):
        raise ValueError("Unknown sub-phenotyping setting.")
    for key, value in changes.items():
        kind = EDITABLE[key]
        if kind == "list":
            if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value) or len(value) > 200:
                raise ValueError(f"Choose the {key.replace('_', ' ')}.")
        elif kind == "mapping":
            if not isinstance(value, dict) or not all(
                isinstance(k, str) and isinstance(v, list) and all(isinstance(m, str) for m in v)
                for k, v in value.items()
            ):
                raise ValueError("Choose the markers of each phenotype.")
        elif kind == "column":
            if not isinstance(value, str) or not COLUMN.fullmatch(value):
                raise ValueError("Choose the phenotype column.")
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
    """Merge *changes* into classic's sub-phenotyping settings file."""
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


def label(value):
    """Classic's safe label for a phenotype (spaces to _, / to -)."""
    return str(value).strip().replace(" ", "_").replace("/", "-") or "NA"


def phenotype_values(table, column):
    from spacemap import dataio

    counts = dataio.load(table, columns=[column])[column].dropna().astype(str).value_counts()
    return [{"name": str(name), "cells": int(cells)} for name, cells in counts.items()]


def _params(chosen):
    method = chosen["method"]
    if method == "Leiden":
        keys = ("resolution", "n_neighbors", "n_pcs")
    elif method == "SOM":
        keys = ("grid", "iterations")
    else:
        keys = ("k_neighbors", "min_cluster_size")
    return {**{key: chosen[key] for key in keys}, "n_meta_clusters": int(chosen["n_meta_clusters"])}


def _discover(frame, chosen, markers, parents, report):
    from spacemap.ph_subpheno_utils import marker_reference_stats, recommend_markers, score_markers

    parent, batch = chosen["parent_col"], chosen["batch_col"] if chosen["batch_col"] in frame.columns else None
    reference = marker_reference_stats(frame, markers, parent)
    found = {}
    groups = frame[parent].astype(str)
    for index, target in enumerate(parents):
        report(
            f"Finding the markers that vary inside {target} ({index + 1} of {len(parents)})…",
            0.2 + 0.6 * index / max(len(parents), 1),
        )
        subset = frame[groups == str(target)]
        if len(subset) < 50:
            found[str(target)] = {
                "cells": int(len(subset)),
                "note": "Too few cells to score (fewer than 50).",
                "markers": [],
            }
            continue
        if len(subset) > DISCOVERY_SAMPLE:
            subset = subset.sample(DISCOVERY_SAMPLE, random_state=42)
        table = recommend_markers(
            score_markers(subset, markers, reference=reference, batch_col=batch, phenotype=str(target)),
            collapsed_below=float(chosen["collapsed_below"]),
            max_batch_eta2=float(chosen["max_batch_eta2"]),
            max_markers=int(chosen["max_markers"]) or None,
        )
        found[str(target)] = {
            "cells": int((groups == str(target)).sum()),
            "markers": [
                {
                    "marker": row.marker,
                    "verdict": row.verdict,
                    "flag": row.flag,
                    "score": round(float(row.score), 4),
                    "spread": round(float(row.sd_ratio), 3) if np.isfinite(row.sd_ratio) else None,
                    "batch_share": round(float(row.batch_eta2), 3) if np.isfinite(row.batch_eta2) else None,
                    "recommended": bool(row.recommended),
                }
                for row in table.itertuples()
            ],
        }
    return found


def run(settings, folder, report):
    """The job: find the markers inside each chosen phenotype, or split each into sub-clusters."""
    from spacemap import dataio
    from spacemap.ph_subpheno_utils import build_feature_matrix, cluster_stability, subcluster, validate_subclusters

    from .leiden import _Log, where
    from .tasks import save_result, validate_table

    chosen = settings["config"]
    table, parent = settings["table"], chosen["parent_col"]
    markers = list(chosen["markers"])
    parents = [str(value) for value in chosen["parent_values"]]
    if not parents:
        raise ValueError("Choose the phenotypes to split.")
    available = dataio.columns(table)
    anchor = settings.get("cell_id_col", "cell_id")
    roi = "batch_id" if "batch_id" in available else None
    batch = chosen["batch_col"] if chosen["batch_col"] in available else None
    wanted = list(dict.fromkeys([anchor, parent, *markers, *[c for c in (batch, roi) if c]]))
    report("Reading the combined table…", 0.03)
    frame = dataio.load(table, columns=wanted)
    frame[markers] = frame[markers].apply(pd.to_numeric, errors="coerce").astype("float32")
    if settings.get("mode") == "discover":
        return {
            "mode": "discover",
            "parent_col": parent,
            "discovery": _discover(frame, chosen, markers, parents, report),
        }
    validate_table(settings)
    groups = frame[parent].astype(str)
    labels = pd.Series(None, index=frame.index, dtype=object)
    blocks, checks, stability, used = [], [], {}, {}
    suggested = None
    for index, target in enumerate(parents):
        keep = [m for m in (chosen["chosen_markers"] or {}).get(target, []) if m in markers]
        if not keep:  # no choice saved: classic's suggestion
            if suggested is None:
                suggested = _discover(frame, chosen, markers, parents, report)
            keep = [row["marker"] for row in suggested.get(target, {}).get("markers", []) if row["recommended"]]
        mask = (groups == target).to_numpy()
        subset = frame[mask]
        report(
            f"Splitting {target}: {len(subset):,} cells on {len(keep)} markers ({index + 1} of {len(parents)})…",
            0.1 + 0.8 * index / len(parents),
        )
        if len(subset) < 100 or len(keep) < 2:
            checks.append(
                {
                    "cluster": f"{label(target)}",
                    "cells": int(len(subset)),
                    "verdict": "skipped",
                    "reason": "too few cells" if len(subset) < 100 else "fewer than two markers kept",
                }
            )
            continue
        matrix = build_feature_matrix(subset, keep, batch_col=batch)
        found = subcluster(
            matrix,
            chosen["method"],
            _params(chosen),
            use_gpu=bool(chosen["use_gpu"]),
            seed=int(chosen["seed"]),
            subsample_size=int(chosen["subsample_size"]),
            logger=_Log(report),
        )
        names = np.array([f"{label(target)}_{int(value)}" for value in found], dtype=object)
        labels.iloc[np.flatnonzero(mask)] = names
        checked = validate_subclusters(found, matrix, keep, batch_labels=subset[batch].to_numpy() if batch else None)
        checked["cluster"] = [f"{label(target)}_{int(value)}" for value in checked.cluster]
        checks.extend(checked.to_dict("records"))
        if chosen["check_stability"]:
            report(f"Checking that {target}'s split reproduces…", None)
            stability[target] = float(
                cluster_stability(
                    matrix,
                    chosen["method"],
                    _params(chosen),
                    n_repeats=3,
                    seed=int(chosen["seed"]),
                    use_gpu=bool(chosen["use_gpu"]),
                    logger=_Log(report),
                )
            )
        used[target] = keep
        # For the heatmap: every sub-cluster over the markers kept for any phenotype, z-scored within the phenotype.
        blocks.append((subset.index, names))
    if labels.notna().sum() == 0:
        raise ValueError("No phenotype could be split: " + "; ".join(f"{c['cluster']}: {c['reason']}" for c in checks))
    union = list(dict.fromkeys(m for keep in used.values() for m in keep))
    display = []
    for rows, names in blocks:
        shown = pd.DataFrame(build_feature_matrix(frame.loc[rows], union, batch_col=batch), columns=union)
        shown["_group"] = names
        display.append(shown)
    from .cluster_profiles import build
    clusters, profile = build(pd.concat(display), union, "_group", name=str, exclude_low=False,
                              label="Kept markers, z-scored within each phenotype")
    verdicts = {str(item["cluster"]): item for item in checks}
    for cluster in clusters:
        verdict = verdicts.get(cluster["name"], {})
        cluster.update(check=verdict.get("verdict", ""), reason=verdict.get("reason", ""))
    output = settings["output_col"]
    columns = {output: labels}
    overwrite = bool(settings.get("overwrite_columns"))
    if overwrite:
        columns[settings["phenotype_col"]] = labels
    saved = save_result(settings, columns, frame[anchor], report)
    kept = sum(1 for item in checks if item.get("verdict") == "keep")
    summary = (
        f"{kept} of {len(checks)} sub-clusters pass every check (size, found in most batches, a separating marker)"
    )
    if stability:
        summary += "; reproducible on re-runs: " + ", ".join(f"{name} {value:.2f}" for name, value in stability.items())
    return {
        **saved,
        "cells": int(labels.notna().sum()),  # the chosen phenotypes' cells, not the whole table
        "mode": "run",
        "clusters": clusters,
        "profile": profile,
        "where": where(labels, frame[roi]) if roi else {},
        "markers": union,
        "markers_used": used,
        "phenotype_column": settings["phenotype_col"],
        "overwrite": overwrite,
        "checks": checks,
        "stability": stability,
        "summary": summary,
    }
