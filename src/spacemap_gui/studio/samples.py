"""Studio's Samples: one row per ROI with its slide and the study's own columns (outcome, arm, timepoint...).

The rows come from the project's lineage (``lineage.py``): every ROI the project
knows, with the slide it was cut from as the ROI step recorded it. The user's
own table (a CSV from the clinic, say) is linked by ROI name or by slide: a
row per slide gives its values to every ROI of that slide. The user chooses
the columns to keep and their names, and can correct any value by hand.

The ROI name is the batch of the combined table's cells, so batch correction
adds these columns to every cell and can correct between slides; later steps
group and compare by them.

Saved as ``<project>/outputs/metadata/samples.csv``, beside the project's other
outputs (never in the code repository).
"""

import io
from pathlib import Path
import re
from collections import OrderedDict

FIXED = ("roi", "slide")
COLUMN = re.compile(r"[A-Za-z][A-Za-z0-9 _.-]{0,63}")
MAX_TEXT = 5_000_000
MAX_ROWS = 50_000
MAX_COLUMNS = 200
_key_counts = OrderedDict()


def path(root):
    return Path(root) / "outputs" / "metadata" / "samples.csv"


def _frame(root):
    import pandas as pd

    file = path(root)
    if not file.is_file():
        return pd.DataFrame(columns=list(FIXED))
    frame = pd.read_csv(file, dtype=str, keep_default_na=False)
    if not set(FIXED) <= set(frame.columns):
        raise ValueError(f"{file.name} lost its roi and slide columns; fix or remove it.")
    return frame


def _write(root, frame):
    file = path(root)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(file)


def _current(root, mappings):
    """The saved rows plus the ROIs new to the project, and the lineage."""
    import pandas as pd

    from . import lineage

    traced = lineage.trace(root, mappings)
    frame = _frame(root)
    # Repair aliases created by earlier builds treating tagged table names as new ROIs.
    from spacemap.viewer.catalog import sample_of
    by_roi = {item["roi"]: item for item in traced["rois"]}
    merged = {}
    for record in frame.to_dict("records"):
        old = record["roi"]
        roi = sample_of(old)
        if roi not in by_roi:
            roi = old
        record["roi"] = roi
        if old != roi and record.get("slide") == old:
            record["slide"] = by_roi[roi]["slide"]
        previous = merged.setdefault(roi, record.copy())
        for column, value in record.items():
            if value and previous.get(column) and value != previous[column]:
                raise ValueError(f"Samples has conflicting {column} values for {roi}; review {path(root)}.")
            if value:
                previous[column] = value
    frame = pd.DataFrame(list(merged.values()), columns=frame.columns).fillna("")
    known = set(frame["roi"])
    new = [{"roi": item["roi"], "slide": item["slide"]} for item in traced["rois"] if item["roi"] not in known]
    if new:
        frame = pd.concat([frame, pd.DataFrame(new)], ignore_index=True).fillna("")
    frame = frame.sort_values("roi", kind="stable").reset_index(drop=True)
    return frame, traced


def table(root, mappings):
    """The samples as rows, with where each slide came from and the slides' source images."""
    frame, traced = _current(root, mappings)
    from .cell_inputs import images
    frame = frame[frame["roi"].isin(images(root, mappings))]
    by_roi = {item["roi"]: item for item in traced["rois"]}
    rows = []
    for record in frame.to_dict("records"):
        item = by_roi.get(record["roi"])
        rows.append(
            {
                **record,
                "_found": item["found"] if item else "gone",
                "_suggested": item["slide"] if item and item["slide"] != record["slide"] else "",
                "_table": item["table"] if item else "",
                "_source": item.get("source", "") if item else "",
            }
        )
    return {
        "columns": list(frame.columns),
        "rows": rows,
        "slides": [{key: value for key, value in slide.items()} for slide in traced["slides"]],
        "saved": path(root).is_file(),
        "file": str(path(root).relative_to(root)),
    }


def _column(name):
    name = str(name or "").strip()
    if not COLUMN.fullmatch(name):
        raise ValueError(
            f"“{name}” cannot be a column name: start with a letter; letters, numbers, space _ . - (up to 64)."
        )
    return name


def edit(root, mappings, roi, column, value):
    """Set one value (a slide corrected by hand, say)."""
    frame, _ = _current(root, mappings)
    if column not in frame.columns:
        raise ValueError(f"There is no column {column}.")
    if column == "roi":
        raise ValueError("The ROI name comes from the project and cannot be changed here.")
    rows = frame.index[frame["roi"] == roi]
    if not len(rows):
        raise ValueError(f"There is no ROI {roi}.")
    text = str(value if value is not None else "").strip()
    if column == "slide" and not text:
        raise ValueError("Every ROI needs a slide.")
    frame.loc[rows, column] = text[:500]
    from . import sample_links
    rules = sample_links.read(root)
    for rule in rules:
        if column in rule["columns"]:
            rule.setdefault("overrides", {}).setdefault(roi, {})[column] = text[:500]
    if rules:
        sample_links.write(root, rules)
    _write(root, frame)
    return table(root, mappings)


def rename(root, mappings, column, new):
    frame, _ = _current(root, mappings)
    new = _column(new)
    if column in FIXED or column not in frame.columns:
        raise ValueError(f"{column} cannot be renamed.")
    if new in frame.columns or new.lower() in FIXED:
        raise ValueError(f"There is a column called {new} already.")
    from . import sample_links
    rules = sample_links.read(root)
    for rule in rules:
        if column in rule["columns"]:
            rule["columns"][new] = rule["columns"].pop(column)
        for override in rule.get("overrides", {}).values():
            if column in override:
                override[new] = override.pop(column)
    if rules:
        sample_links.write(root, rules)
    _write(root, frame.rename(columns={column: new}))
    return table(root, mappings)


def add(root, mappings, column):
    """Add an empty clinical/sample column for manual entry, without requiring a file."""
    frame, _ = _current(root, mappings)
    name = _column(column)
    if name in frame.columns or name.lower() in FIXED:
        raise ValueError(f"There is a column called {name} already.")
    if len(frame.columns) >= MAX_COLUMNS:
        raise ValueError(f"Keep the Samples table under {MAX_COLUMNS} columns.")
    frame[name] = ""
    _write(root, frame)
    return table(root, mappings)


def drop(root, mappings, column):
    frame, _ = _current(root, mappings)
    if column in FIXED or column not in frame.columns:
        raise ValueError(f"{column} cannot be removed.")
    from . import sample_links
    rules = sample_links.read(root)
    if rules:
        sample_links.write(root, sample_links.remove_columns(rules, {column}))
    _write(root, frame.drop(columns=[column]))
    return table(root, mappings)


def _key(value):
    """A name as the project writes it: no extension or step suffix, case and spaces ignored."""
    from spacemap.viewer.catalog import sample_of

    return re.sub(r"\s+", "", sample_of(str(value).strip())).lower()


def read_table(text):
    """The user's table from the text of a CSV or tab-separated file, every value as text."""
    import pandas as pd

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Choose a CSV file with a header row.")
    if len(text) > MAX_TEXT:
        raise ValueError("That file is larger than 5 MB; keep only the columns you need.")
    text = text.lstrip("\ufeff")  # Excel's byte-order mark
    header = text.splitlines()[0]
    separator = max((",", ";", "\t"), key=header.count)  # CSV, European CSV or tab-separated
    if not header.count(separator):
        raise ValueError(
            "The table needs a header row, a matching-key column, and at least one column to add."
        )
    try:
        found = pd.read_csv(io.StringIO(text), sep=separator, dtype=str, keep_default_na=False)
    except (ValueError, pd.errors.ParserError) as error:
        raise ValueError("That file could not be read as a table; in Excel use Save As, CSV.") from error
    found.columns = [str(name).strip() for name in found.columns]
    if found.empty or len(found.columns) < 2:
        raise ValueError(
            "The table needs a header row, a matching-key column, and at least one column to add."
        )
    if len(found) > MAX_ROWS or len(found.columns) > MAX_COLUMNS:
        raise ValueError(f"Keep the table under {MAX_ROWS:,} rows and {MAX_COLUMNS} columns.")
    return found


def suggest_link(root, mappings, text):
    """The file's columns with a few values each, and the column and level that match the project best."""
    found = read_table(text)
    import pandas as pd
    frame = pd.DataFrame(table(root, mappings)["rows"], columns=["roi", "slide"])
    targets = {"roi": {_key(v) for v in frame["roi"]}, "slide": {_key(v) for v in frame["slide"]}}
    best = (0, "", "roi")
    columns = []
    for name in found.columns:
        values = found[name].astype(str)
        keys = {_key(v) for v in values if v.strip()}
        for level, names in targets.items():
            hits = len(keys & names)
            if hits > best[0] or (hits == best[0] and hits and level == "roi" and best[2] != "roi"):
                best = (hits, name, level)
        columns.append({"name": name, "examples": list(dict.fromkeys(v for v in values if v.strip()))[:3]})
    return {"columns": columns, "rows": len(found), "link_column": best[1], "link_to": best[2], "matches": best[0],
            "targets": link_targets(root, mappings)}


def link_targets(root, mappings):
    from . import correction

    frame, _ = _current(root, mappings)
    paths = list(correction.tables(root, mappings).values())
    headers = correction._headers(paths)
    common = [name for name in next(iter(headers.values()), []) if all(name in names for names in headers.values())]
    return ([{"value": "roi", "label": "ROI name"}, {"value": "slide", "label": "Slide ID"},
             {"value": "table", "label": "Cell table filename"}]
            + [{"value": "sample:" + name, "label": "Samples · " + name} for name in frame if name not in FIXED]
            + [{"value": "column:" + name, "label": "Cell tables · " + name} for name in common])


def _counts(source, column):
    """Read only the chosen matching column; cache low-cardinality sample IDs by file identity."""
    from .cell_inputs import read_frame
    from .sample_links import key

    source = Path(source)
    info = source.stat()
    cache_key = (str(source), info.st_size, info.st_mtime_ns, column)
    if cache_key in _key_counts:
        return _key_counts[cache_key]
    values = read_frame(source, columns=[column], text_columns=[column])[column].map(key)
    counts = values.value_counts().to_dict()
    if len(counts) <= 10000:
        _key_counts[cache_key] = counts
        while len(_key_counts) > 128:
            _key_counts.popitem(last=False)
    return counts


def link(root, mappings, text, link_column, link_to, columns, apply=False):
    """Add the chosen columns of the user's table, linked by ROI or slide; *apply* False only previews."""
    import pandas as pd

    found = read_table(text)
    if link_column not in found.columns:
        raise ValueError("Choose the matching column in your imported table.")
    targets = {item["value"] for item in link_targets(root, mappings)}
    if link_to not in targets:
        raise ValueError("Choose the matching column in the dataset.")
    if not isinstance(columns, dict) or not columns:
        raise ValueError("Choose at least one column to add.")
    frame, _ = _current(root, mappings)
    wanted = {}
    for source, name in columns.items():
        if source not in found.columns or source == link_column:
            raise ValueError(f"The file has no column {source} to add.")
        name = _column(name or source)
        if name.lower() in FIXED:
            raise ValueError(f"Name the column {source} something other than {name}: roi and slide are the project's.")
        if name in wanted.values():
            raise ValueError(f"Two columns would both be called {name}.")
        wanted[source] = name
    from . import correction, sample_links
    dataset_column = link_to[7:] if link_to.startswith("column:") else None
    normalize = _key if link_to in (*FIXED, "table") else sample_links.key
    keys = found[link_column].map(normalize)
    repeated = sorted(set(found.loc[keys.duplicated() & (keys != ""), link_column]))
    if repeated:
        raise ValueError(
            f"{len(repeated)} names appear on more than one row of the file ({', '.join(repeated[:5])}); "
            "each matching key needs one row."
        )
    rows = {key: index for key, index in zip(keys, found.index) if key}
    tables = correction.tables(root, mappings)
    active = {row["roi"] for row in table(root, mappings)["rows"]}
    if dataset_column:
        counts = {roi: _counts(source, dataset_column) for roi, source in tables.items()}
        lookups = {name: {key: str(found.at[index, source]) for key, index in rows.items()} for source, name in wanted.items()}
        result = frame.copy()
        used = set()
        matching_rois, matching_cells = set(), 0
        for roi, quantities in counts.items():
            hits = set(quantities) & rows.keys()
            used.update(rows[key] for key in hits)
            if hits:
                matching_rois.add(roi)
                matching_cells += sum(quantities[key] for key in hits)
            for name, lookup in lookups.items():
                values = {lookup.get(key, "") for key in quantities}
                result.loc[result["roi"] == roi, name] = next(iter(values)) if len(values) == 1 else "Varies by cell" if values else ""
        result = result.fillna("")
        preview = result[result["roi"].isin(active)]
        summary = {"matched": len(matching_rois), "matched_cells": int(matching_cells), "rois": len(active),
                   "cells": int(sum(sum(values.values()) for values in counts.values())),
                   "missing": sorted(active - matching_rois)[:50],
                   "unused": [str(found.at[i, link_column]) for i in found.index if i not in used][:50],
                   "replaced": [name for name in wanted.values() if name in frame],
                   "preview": preview[["roi", "slide", *wanted.values()]].head(8).to_dict("records")}
        if not matching_rois:
            raise ValueError(f"No values in {dataset_column} match the file's {link_column} column.")
        if apply:
            rules = sample_links.remove_columns(sample_links.read(root), wanted.values())
            rules.append({"target": dataset_column, "columns": lookups})
            sample_links.write(root, rules)
            _write(root, result)
            summary["table"] = table(root, mappings)
        return summary
    if link_to == "table":
        target_values = frame["roi"].map(lambda roi: Path(tables[roi]).name if roi in tables else "")
    else:
        target_values = frame[link_to[7:] if link_to.startswith("sample:") else link_to]
    matched = target_values.map(lambda value: rows.get(normalize(value))).where(frame["roi"].isin(active))
    used = {int(index) for index in matched.dropna()}
    result = frame.copy()
    for source, name in wanted.items():
        values = matched.map(lambda index, source=source: found.at[int(index), source] if pd.notna(index) else "")
        result[name] = values.where(frame["roi"].isin(active), frame[name] if name in frame else "").fillna("").astype(str)
    replaced = [name for name in wanted.values() if name in frame.columns]
    summary = {
        "matched": int(matched.notna().sum()),
        "rois": len(active),
        "missing": [str(v) for v in frame.loc[matched.isna() & frame["roi"].isin(active), "roi"]][:50],
        "unused": [str(found.at[i, link_column]) for i in found.index if i not in used][:50],
        "replaced": replaced,
        "preview": result.loc[result["roi"].isin(active), ["roi", "slide", *wanted.values()]].head(8).to_dict("records"),
    }
    if not summary["matched"]:
        raise ValueError(f"No {link_to} of the project matches the file's {link_column} column.")
    if apply:
        rules = sample_links.read(root)
        if rules:
            sample_links.write(root, sample_links.remove_columns(rules, wanted.values()))
        _write(root, result)
        summary["table"] = table(root, mappings)
    return summary


def values(root, mappings=None):
    """{ROI: {column: value}}, for the steps that add them to cells: as saved, with *mappings* also the
    ROIs not saved yet (their slide from the lineage), as the page shows them."""
    frame = _current(root, mappings)[0] if mappings is not None else _frame(root)
    return {record.pop("roi"): record for record in frame.to_dict("records")}
