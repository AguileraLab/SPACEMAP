"""Studio's Sub-phenotyping (Tab 7): chosen phenotypes split into functional states, with classic's steps.

Settings are classic's ``outputs/phenotyping/cache/subphenotyping_params.json`` (read and merged); the
steps are classic's (``spacemap.ph_subpheno_utils``), for each chosen phenotype on its own:

1. **Marker review**: the Studio figure workflow shows exact descriptive summaries
   and links inspected cells to Data/image. Explicit choices populate chosen_markers.
   The legacy mode="discover" endpoint remains for compatibility; Studio does not
   present its heuristic scores as a separate decision method.
2. **Run** (a job, ``mode="run"``): all checked panel markers, or the user's explicit per-phenotype
   subset, transformed and normalized using the chosen group (``build_feature_matrix``),
   clustered with Leiden, SOM or VORTEX and
   merged to N (``subcluster``), then checked (``validate_subclusters``: size, reproducible across
   batches, a separating marker; optional stability across re-runs).

The run saves one new column ``<output>_clusters`` ("<Phenotype>_<n>" for the chosen phenotypes'
cells) beside the table, with each group's size, top markers, check, a heatmap and where it is, so the
cluster explorer shows and names them into ``<output>``.
Overwriting updates only successfully clustered cells, preserving other phenotypes,
excluded images/regions and skipped populations in both result columns.
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
    "figure_options": {},
    "transform": "auto",
    "cofactor": 5.0,
    "normalization": "zscore",
    "normalization_col": "",
    "filters": [],
    "reference_column": "",
    "model_id": "",
    "limit_model_phenotypes": False,
    "model_phenotypes": [],
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
    "figure_options": "figures",
    "transform": ("auto", "stored"),
    "cofactor": (float, .001, 65535),
    "normalization": ("zscore", "robust", "none"),
    "normalization_col": "optional_column",
    "filters": "filters",
    "reference_column": "optional_column",
    "model_id": "model",
    "limit_model_phenotypes": bool,
    "model_phenotypes": "list",
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
    merged = {**DEFAULTS, **{key: value for key, value in saved.items() if value is not None and key in DEFAULTS}}
    # Older figure-only transformation choices become the shared transformation.
    for key in ('transform', 'cofactor'):
        if key not in saved and key in (saved.get('figure_options') or {}):
            merged[key] = saved['figure_options'][key]
    return merged


def check(changes):
    from .leiden import COLUMN

    if not isinstance(changes, dict) or set(changes) - set(EDITABLE):
        raise ValueError("Unknown sub-phenotyping setting.")
    for key, value in changes.items():
        kind = EDITABLE[key]
        if kind == "figures":
            from .marker_figures import options
            options(value)
        elif kind == "filters":
            if not isinstance(value, list) or len(value) > 10:
                raise ValueError('Use up to ten cell filters.')
            for item in value:
                if (not isinstance(item, dict) or set(item) != {'column', 'values', 'op'} or
                        not isinstance(item['column'], str) or not item['column'] or
                        item['op'] not in {'in', 'not_in'} or not isinstance(item['values'], list) or
                        not 1 <= len(item['values']) <= 200 or
                        any(not isinstance(v, str) or not v or len(v) > 200 for v in item['values'])):
                    raise ValueError('Choose a column and value for each cell filter.')
        elif kind == "model":
            if not isinstance(value, str) or (value and not re.fullmatch(r'[a-f0-9]{32}', value)):
                raise ValueError('Choose a listed model.')
        elif kind == "optional_column":
            if not isinstance(value, str) or len(value) > 128:
                raise ValueError('Choose a reference column.')
        elif kind == "list":
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


def check_inputs(chosen, available):
    """Validate input roles and keep output columns distinct from every input."""
    check({k: v for k, v in chosen.items() if k in EDITABLE})
    required = [chosen['parent_col'], *chosen['markers'], *(f['column'] for f in chosen.get('filters', []))]
    if chosen.get('reference_column'):
        required.append(chosen['reference_column'])
    if chosen.get('normalization_col') and chosen.get('normalization', 'zscore') != 'none':
        required.append(chosen['normalization_col'])
    missing = set(required) - set(available)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    if not chosen['parent_values']:
        raise ValueError('Choose the phenotypes to split.')
    if len({label(v) for v in chosen['parent_values']}) != len(chosen['parent_values']):
        raise ValueError('These phenotype names produce duplicate subset IDs. Split them in separate runs.')
    if not chosen['markers']:
        raise ValueError('Choose markers for clustering.')
    forbidden = {chosen['parent_col'], chosen.get('reference_column'), chosen.get('normalization_col'), 'cell_id', 'label'}
    if forbidden.intersection(chosen['markers']) or any(m.startswith('phenotype_') for m in chosen['markers']):
        raise ValueError('Cluster on measurements, not phenotype labels or cell IDs.')
    output = chosen['studio_output_col']
    if {output, output + '_clusters'} & set(required):
        raise ValueError('Save the subsets in a new column, separate from the inputs.')
    if chosen.get('model_id'):
        from . import reference_models
        _, meta, artifact = reference_models._source(chosen['model_id'])
        present = set(meta['markers']) & set(available)
        if not present:
            raise ValueError('The model has no matching markers. Choose another model or No model.')
        if len(present) < len(meta['markers']) and not (meta.get('adaptable') and meta.get('allow_missing', True)):
            raise ValueError('This model needs all its markers. Choose another model or No model.')
        if chosen.get('limit_model_phenotypes') and (not chosen.get('model_phenotypes') or
                set(chosen['model_phenotypes']) - set(artifact['model'].classes_)):
            raise ValueError('Choose at least one phenotype from the model.')


def region_mask(frame, filters):
    mask = np.ones(len(frame), bool)
    for rule in filters:
        column = frame[rule['column']]
        if pd.api.types.is_numeric_dtype(column.dtype) and not pd.api.types.is_bool_dtype(column.dtype):
            values = pd.to_numeric(pd.Series(rule['values']), errors='coerce').dropna()
            matched = column.isin(values)
        else:
            matched = column.astype('string').str.strip().isin(rule['values'])
        mask &= (column.notna() & (matched if rule['op'] == 'in' else ~matched)).to_numpy(bool)
    return mask


def image_scope(root, table):
    from .phenotyping_images import sources
    selected = sources(root, table)
    if len(selected) != 1 or Path(selected[0]['table']).resolve() != Path(table).resolve():
        raise ValueError('Choose a merged cell table for these images, or select one image and its cell table.')
    return selected[0]


def selection(table, chosen, image_settings=None):
    from spacemap import dataio
    check({k: chosen[k] for k in ('parent_col', 'parent_values', 'filters') if k in chosen})
    image_settings = image_settings or {}
    roi = image_settings.get('roi_column')
    wanted = list(dict.fromkeys([chosen['parent_col'], *(f['column'] for f in chosen.get('filters', [])), *([roi] if roi else [])]))
    if set(wanted) - set(dataio.columns(table)):
        raise ValueError('Choose existing columns for the cell selection.')
    frame = dataio.load(table, columns=wanted)
    from .phenotyping_images import mask
    selected = mask(frame, image_settings) & region_mask(frame, chosen.get('filters', [])) & frame[chosen['parent_col']].astype('string').isin(chosen['parent_values']).fillna(False).to_numpy(bool)
    return {'cells': int(selected.sum()), 'total': len(frame),
            'parents': {str(k): int(v) for k, v in frame.loc[selected, chosen['parent_col']].value_counts().items()}}


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

    parent = chosen["parent_col"]
    batch = chosen.get("normalization_col") or chosen["batch_col"]
    batch = batch if batch in frame.columns else None
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


def naming_cells(frame, settings):
    """Restrict name saves to the cells actually split by this sub-phenotyping run."""
    scope = settings['subphenotype_run']
    anchor, cluster = settings.get('cell_id_col', 'cell_id'), settings['clusters_col']
    if scope.get('membership_file'):
        path = Path(scope['membership_file'])
        if not path.is_file():
            raise ValueError('This run\'s saved cell membership is missing. Rerun sub-phenotyping before naming it.')
        membership = pd.read_parquet(path, columns=[anchor, cluster])
        ids = membership[anchor].astype(str)
        positions = pd.Index(frame[anchor].astype(str)).get_indexer(ids)
        if ids.duplicated().any() or (positions < 0).any():
            raise ValueError('This run\'s cells have changed. Rerun sub-phenotyping before naming it.')
        selected = frame.iloc[positions]
        if not np.array_equal(selected[cluster].astype(str).to_numpy(), membership[cluster].astype(str).to_numpy()):
            raise ValueError('These subclusters have changed. Name the latest run instead.')
    else:
        # Older saved runs have selection metadata, but no membership snapshot.
        # Limit them by both the original selection and their recorded clusters.
        from spacemap import dataio
        selection = scope['selection']
        parent, filters = selection['parent_column'], selection.get('filters', [])
        columns = list(dict.fromkeys([anchor, parent, *(rule['column'] for rule in filters)]))
        source = dataio.load(settings['table'], columns=columns)
        keep = region_mask(source, filters) & source[parent].astype('string').isin(selection['parents']).fillna(False).to_numpy(bool)
        selected = frame.loc[frame[anchor].astype(str).isin(source.loc[keep, anchor].astype(str))
                             & frame[cluster].astype('string').isin(scope['clusters']).fillna(False)]
    if selected.empty:
        raise ValueError('No cells from this sub-phenotyping run remain to name.')
    return selected


def run(settings, folder, report):
    """The job: find the markers inside each chosen phenotype, or split each into sub-clusters."""
    from spacemap import dataio
    from spacemap.ph_subpheno_utils import build_feature_matrix, cluster_stability, subcluster, validate_subclusters

    from .leiden import _Log, where
    from .tasks import save_result, validate_table
    from .phenotyping_images import mask as image_mask
    from spacemap.marker_preprocessing import transform

    chosen = settings["config"]
    table, parent = settings["table"], chosen["parent_col"]
    markers = list(chosen["markers"])
    parents = [str(value) for value in chosen["parent_values"]]
    if not parents:
        raise ValueError("Choose the phenotypes to split.")
    available = dataio.columns(table)
    check_inputs(chosen, available)
    anchor = settings.get("cell_id_col", "cell_id")
    roi = settings.get('roi_column') or ("batch_id" if "batch_id" in available else None)
    batch = chosen.get("normalization_col") or None
    normalization = chosen.get("normalization", "zscore")
    if normalization == "none":
        batch = None
    qc_batch = batch or (chosen["batch_col"] if chosen["batch_col"] in available else roi)
    wanted = list(dict.fromkeys([anchor, parent, *markers, *[c for c in (batch, roi, qc_batch) if c],
                                *(f['column'] for f in chosen.get('filters', []))]))
    report("Reading selected cells…", 0.03)
    frame = dataio.load(table, columns=wanted)
    images = image_mask(frame, settings)
    scope = images & region_mask(frame, chosen.get('filters', []))
    if not (scope & frame[parent].astype('string').isin(parents).fillna(False).to_numpy(bool)).any():
        raise ValueError('No cells match. Check the phenotype and region filters.')
    # Transform only selected images; source measurements remain untouched.
    frame[markers] = frame[markers].apply(pd.to_numeric, errors='coerce').astype(float)
    frame.loc[images, markers] = transform(frame.loc[images], markers, chosen.get('transform', 'auto'), chosen.get('cofactor', 5.))
    analysis = frame.loc[scope]
    preprocessing = {k: chosen.get(k, DEFAULTS[k]) for k in ('transform', 'cofactor', 'normalization', 'normalization_col')}
    preprocessing.update(corrected_kept=[m for m in markers if m.endswith('_corr')],
                         imputation='Missing normalized values use the group center; unscaled values use the group median.',
                         clip=5 if normalization != 'none' else None)
    if settings.get("mode") == "discover":
        return {
            "mode": "discover",
            "parent_col": parent,
            "discovery": _discover(analysis, chosen, markers, parents, report),
            "preprocessing": preprocessing,
            "selected_images": settings.get('selected_images'),
        }
    validate_table(settings)
    groups = frame[parent].astype(str)
    labels = pd.Series(None, index=frame.index, dtype=object)
    blocks, checks, stability, used, fitted = [], [], {}, {}, {}
    for index, target in enumerate(parents):
        # Run honors the checked panel. Discovery is advisory and never runs
        # implicitly: only an explicit per-phenotype choice can narrow the inputs.
        keep = [m for m in (chosen["chosen_markers"] or {}).get(target, markers) if m in markers]
        mask = (groups == target).to_numpy() & scope
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
                    "reason": f"needs 100 cells ({len(subset)} selected)" if len(subset) < 100 else "fewer than two markers kept",
                }
            )
            continue
        matrix, fitted[target] = build_feature_matrix(subset, keep, batch_col=batch, normalization=normalization, return_info=True)
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
        checked = validate_subclusters(found, matrix, keep, batch_labels=subset[qc_batch].to_numpy() if qc_batch else None)
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
        shown = pd.DataFrame(build_feature_matrix(frame.loc[rows], union, batch_col=batch, normalization=normalization), columns=union)
        shown["_group"] = names
        display.append(shown)
    from .cluster_profiles import build
    transform_label = "Arcsinh raw; corrected kept" if preprocessing['transform'] == 'auto' else "Stored values"
    norm_label = {'zscore': 'Z-score', 'robust': 'Median/IQR', 'none': 'No normalization'}[normalization]
    clusters, profile = build(pd.concat(display), union, "_group", name=str, exclude_low=False,
                              label=f"{transform_label} → {norm_label}, within each phenotype" + (f" and {batch}" if batch else ""))
    verdicts = {str(item["cluster"]): item for item in checks}
    for cluster in clusters:
        verdict = verdicts.get(cluster["name"], {})
        cluster.update(check=verdict.get("verdict", ""), reason=verdict.get("reason", ""))
    output = settings["output_col"]
    from .subphenotype_guidance import build as guidance
    hints = guidance(table, labels, chosen, report)
    processed = labels.notna()
    columns = {output: labels.loc[processed]}
    overwrite = bool(settings.get("overwrite_columns"))
    if overwrite:
        columns[settings["phenotype_col"]] = labels.loc[processed]
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    (folder / 'preprocessing.json').write_text(json.dumps({**preprocessing, 'fitted': fitted}, indent=2, allow_nan=False))
    # Naming must use this exact membership, including the image/region filters,
    # rather than remapping the shared column for other runs' cells.
    membership = frame.loc[processed, [anchor]].copy()
    membership[output] = labels.loc[processed]
    membership.to_parquet(folder / 'subphenotype_membership.parquet', index=False)
    saved = save_result({**settings, 'row_scoped': True}, columns, frame.loc[processed, anchor], report)
    kept = sum(1 for item in checks if item.get("verdict") == "keep")
    summary = f"{kept} of {len(checks)} subsets pass the cluster checks"
    if stability:
        summary += "; stability (ARI): " + ", ".join(f"{name} {value:.2f}" for name, value in stability.items())
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
        "scoped_membership": True,
        "checks": checks,
        "stability": stability,
        "summary": summary,
        "guidance": hints,
        "preprocessing": preprocessing,
        "selection": {'parent_column': parent, 'parents': parents, 'filters': chosen.get('filters', []),
                      'images': settings.get('selected_images'), 'image_column': roi},
    }
