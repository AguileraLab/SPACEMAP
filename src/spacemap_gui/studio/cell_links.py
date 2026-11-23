"""Link measurements to a segmentation by ROI and local cell label."""

from pathlib import Path


def table_layer(layer, table_id):
    """Different tables on one cell layer may use different identity column names."""
    return {**layer, **(layer.get("table_roles", {}).get(table_id) or {})}


def bind(root, table, token, *, prefix="merged_", expected_image=None):
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
    values = data_view.saved_image_values(table, roles["roi"])
    image_roi = layer["roi"]
    candidates = [value for value in values if sample_of(value) == sample_of(image_roi)]
    if not roles["roi"]:
        same_image = (Path(layer.get('image', '')).resolve() == Path(expected_image).resolve()) if expected_image else (
            sample_of(image_roi) == phenotyping_inputs.single_image(root, table))
        if not same_image:
            return {"linked": False, "reason": "This table belongs to a different image. Check Column roles.", "roi": image_roi}
        roi = image_roi
    elif image_roi in values:
        roi = image_roi
    elif len(candidates) == 1:
        roi = candidates[0]
    else:
        return {"linked": False, "reason": "Cannot match this image uniquely. Check the Image column in Column roles.", "roi": image_roi}
    tables = dict(layer.get("tables") or {})
    reference = layer.get("reference_table")
    if not reference:
        reference = next((path for path in tables.values() if Path(path).resolve() != Path(table).resolve()), None)
    pyramid = layer.get("pyramid")
    count = cells.pyramid_cell_count(pyramid) if pyramid and cells.pyramid_ready(pyramid) else None
    fits, reason = cell_view.matches(reference, table, roi, count, roles["roi"], label, reference_roi=image_roi)
    if not fits:
        return {"linked": False, "reason": reason, "roi": layer["roi"], "retryable": fits is None}
    keys = {"roi":roi, "roi_column": roles["roi"], "label_column": label, "cell_id_column": roles["cell_id"]}
    import hashlib
    import json
    key = prefix + hashlib.sha256(json.dumps([str(table), keys], sort_keys=True).encode()).hexdigest()[:12]
    table_roles = {**layer.get("table_roles", {}), key: keys}
    registry.connect_cell_table(token, key, table, keys)
    return {"linked": True, "token": token, "table": key, "keys": keys, "table_roles": table_roles,
            "aliases": [name for name, path in tables.items() if Path(path).resolve() == Path(table).resolve()],
            "roi": roi, "reason": reason}


def context(channel, tokens=()):
    from spacemap import dataio
    from spacemap.viewer.cells import _edits_token
    from spacemap.viewer.sources import fingerprint
    from spacemap_gui.viewer import registry
    from . import phenotyping_inputs
    from .data_view import Preparing

    project = registry.session_project(channel) or {}
    table = project.get("unified")
    if not table or not Path(table).is_file():
        return {"table": None, "layers": {}}
    # A file/sidecar stamp is enough to distinguish saved results, including an
    # overwrite of an existing column. Never scan cell values for this check.
    revision = f"{fingerprint(table)}|{_edits_token(str(table))}"
    root = project["root"]
    roles = {**phenotyping_inputs.column_roles(root, table), "label": phenotyping_inputs.segmentation_column(root, table)}
    links = {}
    for token in dict.fromkeys(tokens):
        try:
            links[token] = bind(root, table, token)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            # A stale/missing cell layer must not hide a readable project table.
            links[token] = {"linked": False, "reason": str(exc), "retryable": isinstance(exc, Preparing)}
    return {"table": {"name": Path(table).name, "path": str(table), "roles": roles,
                      "columns": dataio.columns(table), "revision": revision}, "layers": links}
