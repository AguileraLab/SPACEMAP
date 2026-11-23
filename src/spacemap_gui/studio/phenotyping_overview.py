"""The Phenotyping step's top section: the combined table it works on, its cells per image and its phenotype columns."""

from pathlib import Path

from . import summary_cache


def overview(table, roi_column="batch_id", slide_column=None, single_roi=""):
    """{"table", "cells", "rois": [{"roi", "cells"}], "phenotypes": [columns]}; counts kept until the table changes."""
    from spacemap import dataio

    if not table or not Path(table).is_file():
        return {"table": "", "cells": 0, "rois": [], "phenotypes": []}
    columns = dataio.columns(str(table))
    def build():
        counts, slides = {}, {}
        if roi_column in columns:
            wanted = list(dict.fromkeys([roi_column, *([slide_column] if slide_column in columns else [])]))
            frame = dataio.load(str(table), columns=wanted, build_fast_copy=False)
            values = frame[roi_column].astype(str)
            counts = values.value_counts().sort_index().to_dict()
            if slide_column in columns:
                slides = frame.groupby(values)[slide_column].agg(lambda x: str(x.iloc[0]) if x.nunique(dropna=False) == 1 else "Multiple").to_dict()
        elif not roi_column and columns:
            wanted = slide_column if slide_column in columns else columns[0]
            parquet = Path(table) if str(table).endswith('.parquet') else None
            if parquet is None:
                twin = dataio.fast_copy_info(table)
                if twin['current']:
                    parquet = Path(twin['twin'])
            if parquet and wanted != slide_column:
                import pyarrow.parquet as pq
                counts[single_roi] = pq.ParquetFile(parquet).metadata.num_rows
            else:
                frame = dataio.load(str(table), columns=[wanted], build_fast_copy=False)
                counts[single_roi] = len(frame)
                if wanted == slide_column and len(frame):
                    slides[single_roi] = str(frame[wanted].iloc[0]) if frame[wanted].nunique(dropna=False) == 1 else "Multiple"
        phenotypes = [c for c in columns if c.startswith("phenotype_") or c in ("consensus", "cluster", "leiden")]
        return {"table": Path(table).name, "cells": int(sum(counts.values())),
                      "rois": [{"roi": roi, "cells": int(n), "slide": slides.get(roi, "")} for roi, n in counts.items()], "phenotypes": phenotypes}
    return summary_cache.cached('phenotyping-overview', [str(table), roi_column, slide_column, single_roi, columns],
                                lambda: summary_cache.table_stamp(table), build)


def results(table, suffix="_cn", roi_column="batch_id", extra_columns=(), single_roi=""):
    """{column: {image: cells with a value}} for the table's result columns ending in *suffix* (neighbourhoods:
    "_cn"), so the Spatial analysis list can say which images each run covered."""
    from spacemap import dataio

    if not table or not Path(table).is_file():
        return {}
    columns = dataio.columns(str(table))  # with the result columns saved beside the table
    names = [c for c in columns if c.endswith(suffix)] if suffix is not None else [
        c for c in columns if c.startswith("phenotype_") or c in {"consensus", "cluster", "leiden"}]
    names = list(dict.fromkeys([*names, *(c for c in extra_columns if c in columns)]))
    def build():
        found = {}
        if names and (not roi_column or roi_column in columns):
            import numpy as np
            import pandas as pd
            frame = dataio.load(str(table), columns=[*([roi_column] if roi_column else []), *names], build_fast_copy=False)
            images = frame[roi_column].astype(str) if roi_column else pd.Series(single_roi, index=frame.index)
            image_codes, image_names = pd.factorize(images)
            for name in names:
                # Test each distinct value once, rather than stripping millions
                # of identical phenotype names on every opening of the tab.
                codes, values = pd.factorize(frame[name])
                meaningful = np.array([bool(str(value).strip()) for value in values] + [False])
                valid = meaningful[codes]  # -1 (missing) indexes the final False
                counts = np.bincount(image_codes[valid], minlength=len(image_names))
                found[name] = {str(image): int(n) for image, n in sorted(zip(image_names, counts)) if n}
        return found
    return summary_cache.cached('phenotyping-results', [str(table), roi_column, single_roi, names],
                                lambda: summary_cache.table_stamp(table), build)


def _job_outputs(root):
    """Cache compact provenance per job; never reparse unchanged full profiles."""
    import json
    from . import jobs
    records = []
    for path in jobs.root(root).glob('*/status.json'):
        request_path = path.parent / 'request.json'
        def stamp():
            return [summary_cache.file_stamp(path), summary_cache.file_stamp(request_path)]
        def build():
            status = json.loads(path.read_text())
            if status.get('status') != 'done':
                return {}
            request = json.loads(request_path.read_text())
            result, config = status.get('result') or {}, request.get('settings') or {}
            return {'kind': request.get('kind'), 'source': result.get('table') or config.get('table'),
                    'names': [result.get('phenotype_column'), config.get('phenotype_col'),
                              result.get('column'), config.get('output_col')]}
        try:
            record = summary_cache.cached('job-phenotyping-output', str(path), stamp, build)
            if record:
                records.append((path.stat().st_mtime_ns, record))
        except (OSError, ValueError, TypeError):
            continue
    return [record for _, record in sorted(records, key=lambda item: item[0], reverse=True)]


def method_columns(root, table, available):
    """Saved-job provenance first; classic/default names are a fallback only."""
    from . import settings
    from spacemap.viewer import reference

    definitions = [("leiden", "L", "Leiden"), ("som", "S", "SOM"), ("vortex", "V", "Vortex"),
                   ("resolve", "T", "Threshold / RESOLVE"), ("consensus", "C", "Consensus ML"),
                   ("mr_ml", "M", "MR-ML"), ("reference", "R", "Manual reference")]
    found = {key: [] for key, _, _ in definitions}
    aliases = {"resolve_all": "resolve", "consensus_ml": "consensus"}
    owned = {}
    for record in _job_outputs(root):
        try:
            kind = aliases.get(record['kind'], record['kind'])
            source = record['source']
            if kind not in found or not source or Path(source).resolve() != Path(table).resolve():
                continue
            for name in record['names']:
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
