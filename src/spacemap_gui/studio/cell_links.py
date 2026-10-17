"""Link the combined measurements to a segmentation by ROI and local cell label."""

from pathlib import Path


def table_layer(layer, table_id):
    """Different tables on one cell layer may use different identity column names."""
    return {**layer, **(layer.get("table_roles", {}).get(table_id) or {})}


def bind(root, table, token):
    from spacemap.viewer import cells
    from spacemap_gui.viewer import registry
    from . import cell_view, phenotyping_inputs

    layer = registry.resolve_object(token, "cells")
    if not layer:
        raise ValueError("Reopen the cell layer to link its measurements.")
    roles = phenotyping_inputs.column_roles(root, table)
    label = phenotyping_inputs.segmentation_column(root, table)
    from . import data_view
    from spacemap.viewer.catalog import sample_of
    values = data_view.info(table, roles["roi"])["values"] if roles["roi"] else []
    image_roi = layer["roi"]
    candidates = [value for value in values if sample_of(value) == sample_of(image_roi)]
    if image_roi in values:
        roi = image_roi
    elif len(candidates) == 1:
        roi = candidates[0]
    else:
        return {"linked": False, "reason": "The ROI column does not identify this image uniquely", "roi": image_roi}
    tables = dict(layer.get("tables") or {})
    reference = layer.get("reference_table")
    if not reference:
        reference = next((path for path in tables.values() if Path(path).resolve() != Path(table).resolve()), None)
    pyramid = layer.get("pyramid")
    count = cells.pyramid_cell_count(pyramid) if pyramid and cells.pyramid_ready(pyramid) else None
    fits, reason = cell_view.matches(reference, table, roi, count, roles["roi"], label, reference_roi=image_roi)
    if not fits:
        return {"linked": False, "reason": reason, "roi": layer["roi"]}
    keys = {"roi":roi, "roi_column": roles["roi"], "label_column": label, "cell_id_column": roles["cell_id"]}
    import hashlib
    import json
    key = "merged_" + hashlib.sha256(json.dumps([str(table), keys], sort_keys=True).encode()).hexdigest()[:12]
    table_roles = {**layer.get("table_roles", {}), key: keys}
    registry.connect_cell_table(token, key, table, keys)
    return {"linked": True, "token": token, "table": key, "keys": keys, "table_roles": table_roles,
            "aliases": [name for name, path in tables.items() if Path(path).resolve() == Path(table).resolve()],
            "roi": roi, "reason": reason}


def context(channel, tokens=()):
    from spacemap_gui.viewer import registry
    from . import phenotyping_inputs

    project = registry.session_project(channel) or {}
    table = project.get("unified")
    if not table or not Path(table).is_file():
        return {"table": None, "layers": {}}
    root = project["root"]
    roles = {**phenotyping_inputs.column_roles(root, table), "label": phenotyping_inputs.segmentation_column(root, table)}
    return {"table": {"name": Path(table).name, "path": str(table), "roles": roles},
            "layers": {token: bind(root, table, token) for token in dict.fromkeys(tokens)}}
