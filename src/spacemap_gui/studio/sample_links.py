"""Saved metadata joins against columns in the per-ROI cell tables."""

import json
import re
from pathlib import Path


def path(root):
    return Path(root) / "outputs/metadata/sample_links.json"


def read(root):
    source = path(root)
    return json.loads(source.read_text()) if source.is_file() else []


def write(root, rules):
    target = path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rules, indent=2))
    temporary.replace(target)


def key(value):
    import pandas as pd
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.0+", text):
        return text.split('.')[0]
    return text


def apply(frame, roi, rules, matching_keys=None):
    """Many cells may share a metadata key; the lookup never multiplies cell rows."""
    for rule in rules:
        target = rule["target"]
        if target not in frame:
            raise ValueError(f"The metadata matching column {target} is missing from {roi}.")
        keys = ((matching_keys or {}).get(target, frame[target])).map(key)
        for column, lookup in rule["columns"].items():
            frame[column] = keys.map(lookup).fillna("")
            override = rule.get("overrides", {}).get(roi, {})
            if column in override:
                frame[column] = override[column]
    return frame


def remove_columns(rules, columns):
    result = []
    for rule in rules:
        rule = {**rule, "columns": {name: values for name, values in rule["columns"].items() if name not in columns}}
        if rule["columns"]:
            result.append(rule)
    return result
