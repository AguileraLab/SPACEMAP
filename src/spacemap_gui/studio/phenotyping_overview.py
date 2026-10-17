"""The Phenotyping step's top section: the combined table it works on, its cells per image and its phenotype columns."""

from pathlib import Path

_SEEN = {}


def overview(table, roi_column="batch_id", slide_column=None):
    """{"table", "cells", "rois": [{"roi", "cells"}], "phenotypes": [columns]}; counts kept until the table changes."""
    from spacemap import dataio
    from spacemap.viewer.sources import fingerprint

    if not table or not Path(table).is_file():
        return {"table": "", "cells": 0, "rois": [], "phenotypes": []}
    columns = dataio.columns(str(table))
    key = (str(table), fingerprint(table), roi_column, slide_column, tuple(columns))
    if key not in _SEEN:
        counts, slides = {}, {}
        if roi_column in columns:
            wanted = list(dict.fromkeys([roi_column, *([slide_column] if slide_column in columns else [])]))
            frame = dataio.load(str(table), columns=wanted)
            values = frame[roi_column].astype(str)
            counts = values.value_counts().sort_index().to_dict()
            if slide_column in columns:
                slides = frame.groupby(values)[slide_column].agg(lambda x: str(x.iloc[0]) if x.nunique(dropna=False) == 1 else "Multiple").to_dict()
        phenotypes = [c for c in columns if c.startswith("phenotype_") or c in ("consensus", "cluster", "leiden")]
        if len(_SEEN) > 4:
            _SEEN.clear()
        _SEEN[key] = {"table": Path(table).name, "cells": int(sum(counts.values())),
                      "rois": [{"roi": roi, "cells": int(n), "slide": slides.get(roi, "")} for roi, n in counts.items()], "phenotypes": phenotypes}
    return _SEEN[key]


_RESULTS = {}


def results(table, suffix="_cn", roi_column="batch_id", extra_columns=()):
    """{column: {image: cells with a value}} for the table's result columns ending in *suffix* (neighbourhoods:
    "_cn"), so the Spatial analysis list can say which images each run covered."""
    from spacemap import dataio
    from spacemap.viewer.sources import fingerprint
    from pandas.api.types import is_numeric_dtype

    if not table or not Path(table).is_file():
        return {}
    columns = dataio.columns(str(table))  # with the result columns saved beside the table
    from spacemap import labels, pending_columns
    dependencies = [labels.sidecar_path(table), *(Path(item["file"]) for item in pending_columns.pending(table))]
    key = (str(table), fingerprint(table), suffix, roi_column, tuple(columns), tuple(extra_columns),
           tuple((str(path), fingerprint(path)) for path in dependencies if path.is_file()))
    if key not in _RESULTS:
        found = {}
        names = [c for c in columns if c.endswith(suffix)] if suffix is not None else [
            c for c in columns if c.startswith("phenotype_") or c in {"consensus", "cluster", "leiden"}]
        names = list(dict.fromkeys([*names, *(c for c in extra_columns if c in columns)]))
        if names and roi_column in columns:
            frame = dataio.load(str(table), columns=[roi_column, *names])
            images = frame[roi_column].astype(str)
            for name in names:
                valid = frame[name].notna()
                if not is_numeric_dtype(frame[name]):
                    valid &= frame[name].astype(str).str.strip().ne("")
                found[name] = images[valid].value_counts().sort_index().astype(int).to_dict()
        if len(_RESULTS) > 8:
            _RESULTS.clear()
        _RESULTS[key] = found
    return _RESULTS[key]


def method_columns(root, table, available):
    """Saved-job provenance first; classic/default names are a fallback only."""
    import json
    from . import jobs, settings
    from spacemap.viewer import reference

    definitions = [("leiden", "L", "Leiden"), ("som", "S", "SOM"), ("vortex", "V", "Vortex"),
                   ("resolve", "T", "Threshold / RESOLVE"), ("consensus", "C", "Consensus ML"),
                   ("mr_ml", "M", "MR-ML"), ("reference", "R", "Manual reference")]
    found = {key: [] for key, _, _ in definitions}
    aliases = {"resolve_all": "resolve", "consensus_ml": "consensus"}
    owned = {}
    folder = jobs.root(root)
    # Read only the small manifests, never job profiles or full cell tables.
    for path in sorted(folder.glob("*/status.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            status = json.loads(path.read_text())
            if status.get("status") != "done":
                continue
            request = json.loads((path.parent / "request.json").read_text())
            kind = aliases.get(request.get("kind"), request.get("kind"))
            result = status.get("result") or {}
            config = request.get("settings") or {}
            source = result.get("table") or config.get("table")
            if kind not in found or not source or Path(source).resolve() != Path(table).resolve():
                continue
            for name in [result.get("phenotype_column"), config.get("phenotype_col"), result.get("column"), config.get("output_col")]:
                if name in available and name not in owned:
                    found[kind].append(name); owned[name] = kind
        except (OSError, ValueError, TypeError):
            continue
    manual = reference.columns(table) if table else []
    for name in available:
        low = name.lower()
        if name in manual:
            kind = "reference"
        elif any(low.endswith(s) for s in ("_score", "_confidence", "_probability")):
            continue
        elif "leiden" in low:
            kind = "leiden"
        elif "vortex" in low:
            kind = "vortex"
        elif "som" in low and low.startswith("phenotype_"):
            kind = "som"
        elif "consensus" in low:
            kind = "consensus"
        elif "mr_ml" in low or "ml_mr" in low:
            kind = "mr_ml"
        elif "resolve" in low or low.startswith("phenotype_threshold"):
            kind = "resolve"
        else:
            continue
        if name not in owned or kind == "reference":
            if name not in found[kind]:
                found[kind].append(name)
    # Custom classic RESOLVE output has no Studio job manifest.
    output = settings.resolve_values(root).get("output_col")
    if output in available and output not in owned and output not in found["resolve"]:
        found["resolve"].append(output)
    return [{"id": key, "letter": letter, "name": name, "columns": found[key]} for key, letter, name in definitions]
