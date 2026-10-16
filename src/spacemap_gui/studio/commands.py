"""One checked executor for Studio controls and assistant viewer requests.

Project names and cell identities come from the registered session. View edits
are returned as lightweight effects as well as a canonical saved view; clients
can apply them without reopening the image. Receipt IDs make retries safe.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

import pandas as pd

from spacemap import dataio, userpaths
from spacemap.viewer import cells, display, live_resolve, reference
from spacemap.viewer.sources import fingerprint, image_info
from spacemap_gui.viewer import registry
from .names import NameChoice, match, normalise
from .schemas import COMMANDS, catalogue, validate

_LOCKS = OrderedDict()
_LOCK = threading.RLock()
_LOGGER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-log")


def _lock(channel):
    with _LOCK:
        if channel not in _LOCKS:
            _LOCKS[channel] = threading.RLock()
        return _LOCKS[channel]


@contextmanager
def _database():
    path = userpaths.state_dir() / "studio_commands.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute(
        "CREATE TABLE IF NOT EXISTS commands(channel TEXT,id TEXT,signature TEXT,result TEXT,undo TEXT,undone INTEGER DEFAULT 0,PRIMARY KEY(channel,id))"
    )
    try:
        with db:
            yield db
    finally:
        db.close()


def _signature(name, args):
    return hashlib.sha256(json.dumps([name, args], sort_keys=True).encode()).hexdigest()


def _receipt(channel, request_id, signature):
    with _database() as db:
        row = db.execute(
            "SELECT signature,result FROM commands WHERE channel=? AND id=?", (channel, request_id)
        ).fetchone()
    if row:
        if row[0] != signature:
            raise ValueError("That request number was already used for another action.")
        return json.loads(row[1])


def _record(channel, request_id, signature, result, undo):
    with _database() as db:
        db.execute(
            "INSERT INTO commands(channel,id,signature,result,undo) VALUES(?,?,?,?,?)",
            (channel, request_id, signature, json.dumps(result), json.dumps(undo)),
        )


def _journal(project, channel, name, args, result, started, elapsed, cpu):
    root = (
        Path(project["root"])
        if project.get("root") and not project.get("read_only")
        else userpaths.logs_dir() / "studio"
    )
    folder = root / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    event = {
        "time": started.isoformat(),
        "channel": channel,
        "name": name,
        "args": args,
        "status": result["status"],
        "command_id": result.get("command_id"),
        "message": result.get("message", ""),
        "duration_s": elapsed,
        "cpu_time_s": cpu,
    }
    import fcntl

    with (folder / "studio_commands.jsonl").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def csv_log():
        from spacemap_gui.run_log import record_run

        record_run(
            root,
            "studio",
            name,
            started_at=started,
            usage={"duration_s": elapsed, "cpu_time_s": cpu},
            status=result["status"],
            error=result.get("message", "") if result["status"] == "error" else "",
            settings=args,
            details={"command_id": result.get("command_id"), "corrections": result.get("corrections", [])},
        )

    _LOGGER.submit(csv_log)


def _image_id(channel, value, corrections):
    from spacemap_gui.viewer.ui import window_images

    choices = window_images(channel)
    wanted = normalise(value)
    exact = [
        entry
        for entry in choices
        if any(
            normalise(alias) == wanted
            for alias in (entry["name"], entry.get("sample", ""), entry["group"] + " | " + entry["name"])
        )
    ]
    if len(exact) == 1:
        found = exact[0]
    elif len(exact) > 1:
        raise NameChoice("image", value, [entry["group"] + " | " + entry["name"] for entry in exact])
    else:
        names = [entry["name"] for entry in choices]
        name = match(value, names, "image", corrections=corrections)
        exact = [entry for entry in choices if entry["name"] == name]
        if len(exact) != 1:
            raise NameChoice("image", value, [entry["group"] + " | " + entry["name"] for entry in exact])
        found = exact[0]
    if value != found["name"]:
        corrections.append({"kind": "image", "requested": value, "matched": found["group"] + " | " + found["name"]})
    return found["id"]


def _view(channel, spec, args, corrections):
    if args.get("image"):
        from spacemap_gui.viewer.ui import window_image_spec

        image_id = _image_id(channel, args["image"], corrections)
        selected = registry.session_images(channel)[image_id]
        active = next((layer for layer in spec.get("layers", []) if layer["type"] == "image"), {})
        current = registry.resolve(active.get("token", ""))
        if not current or Path(current).resolve() != Path(selected["path"]).resolve():
            spec = window_image_spec(channel, image_id)
    image = next((layer for layer in spec.get("layers", []) if layer["type"] == "image"), None)
    if not image:
        raise ValueError("Open an image first.")
    path = registry.resolve(image["token"])
    if not path:
        raise ValueError("Reopen this image to reconnect its layers.")
    return spec, image, image_info(path)


def _cell_table(spec, image):
    layer = next(
        (layer for layer in spec["layers"] if layer["type"] == "cells" and layer.get("image_token") == image["token"]),
        None,
    )
    if not layer:
        raise ValueError("This image has no cell layer. Choose an ROI with cells.")
    registered = registry.resolve_object(layer["token"], "cells")
    if not registered:
        raise ValueError("Reopen this image to reconnect its cells.")
    tables = registered.get("tables") or {}
    table_id = layer.get("table") or next(iter(tables), "")
    from .cell_links import table_layer
    registered = table_layer(registered, table_id)
    table = tables.get(table_id)
    if not table:
        raise ValueError("Choose a cell table first.")
    layer["table"] = table_id
    return layer, registered, table


def _keys(layer):
    return {"roi_column": layer.get("roi_column", "batch_id"), "label_column": layer.get("label_column", "label")}


def _column(table, layer, value, corrections):
    columns = dataio.columns(table)
    default = layer.get("color_by") or (
        "phenotype_ml_consensus"
        if "phenotype_ml_consensus" in columns
        else next((c for c in columns if c.startswith("phenotype_")), "")
    )
    return match(value or default, columns, "column", corrections=corrections)


def _point_density(channel, spec, layer, registered, image, args, corrections, request_id, undo):
    """A density map of one group of a point table (Tab 8's triads, say) in this ROI.

    The tables the cell layer carries, and the project's triad tables saved since the image was opened.
    """
    from spacemap.viewer.catalog import find_point_tables

    own = {item["id"] for item in layer.get("tables", [])}
    options = {item["name"]: item["id"] for item in layer.get("density_tables", []) if item["id"] not in own}
    root = (registry.session_project(channel) or {}).get("root")
    saved = dict(find_point_tables(root)) if root else {}
    if not options and not saved:
        raise ValueError("This project has no triad tables yet (classic Tab 8, Triads, saves them).")
    name = match(args["table"], list(dict.fromkeys([*options, *saved])), "table", corrections=corrections)
    tables = registered.get("tables") or {}
    token, table_id = layer["token"], options.get(name)
    if table_id is None or table_id not in tables:
        # Saved after the image was opened: the cells are registered again with this table too.
        table_id = "p" + str(len(tables))
        tables = {**tables, table_id: saved[name]}
        token = registry.register_object(
            "cells", **{**{k: v for k, v in registered.items() if k != "kind"}, "tables": tables}
        )
    path = tables[table_id]
    column = args.get("column") or "triad_label"
    groups = dict(cells.group_counts(path, registered["roi"], column))
    if not groups:
        raise ValueError(f"{name} has nothing in {registered['roi']}.")
    chosen = match(args["phenotype"], list(groups), "phenotype", corrections=corrections)
    density = {
        "id": "density-" + request_id,
        "type": "density",
        "token": token,
        "image_token": image["token"],
        "name": chosen + " density",
        "visible": True,
        "opacity": 0.65,
        "table": table_id,
        "tables": [
            *layer.get("density_tables", []),
            *([{"id": table_id, "name": name}] if table_id not in options.values() else []),
        ],
        "fcol": column,
        "fvals": [chosen],
        "radius": args["radius"],
        "cmap": "magma",
        "open": True,
    }
    spec["layers"].append(density)
    return (
        spec,
        [{"type": "add_layer", "layer": density}],
        {"count": groups[chosen]},
        undo,
        f"Showing where {chosen} is ({groups[chosen]:,} in this ROI).",
    )


def _groups(table, registered, column):
    ids, values = cells.roi_values(table, registered["roi"], column, **_keys(registered))
    counts = pd.Series(values).fillna("Unassigned").astype(str).replace({"": "Unassigned"}).value_counts()
    return counts.to_dict()


def _snapshot(spec, client_view):
    """Accept display preferences only for layers already registered in this view."""
    if not client_view:
        return spec
    if not isinstance(client_view, dict) or client_view.get("id") != spec.get("id"):
        raise ValueError("The image changed. Try the action again in the current view.")
    incoming = client_view.get("layers") or []
    if not isinstance(incoming, list) or len(incoming) > 200:
        raise ValueError("The view settings are not valid.")
    by_id = {layer["id"]: layer for layer in spec.get("layers", [])}
    for item in incoming:
        if not isinstance(item, dict) or item.get("id") not in by_id:
            continue
        layer = by_id[item["id"]]
        if isinstance(item.get("visible"), bool):
            layer["visible"] = item["visible"]
        if (
            isinstance(item.get("opacity"), (int, float))
            and math.isfinite(item["opacity"])
            and 0 <= item["opacity"] <= 1
        ):
            layer["opacity"] = item["opacity"]
        if layer["type"] == "image" and isinstance(item.get("channels"), list):
            channels = display.clean(item["channels"])
            size = image_info(registry.resolve(layer["token"])).channel_count
            layer["channels"] = [c for c in channels if 0 <= c["index"] < size]
        if layer["type"] == "cells":
            registered = registry.resolve_object(layer["token"], "cells") or {}
            if item.get("table") in registered.get("tables", {}):
                layer["table"] = item["table"]
            for key in ("color_by", "mode"):
                if isinstance(item.get(key), str) and len(item[key]) < 512:
                    layer[key] = item[key]
            if isinstance(item.get("hidden"), list) and all(isinstance(v, str) for v in item["hidden"]):
                layer["hidden"] = item["hidden"][:1000]
    bounds = client_view.get("bounds")
    if (
        isinstance(bounds, list)
        and len(bounds) == 4
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds)
        and bounds[2] > 0
        and bounds[3] > 0
    ):
        spec["bounds"] = bounds
    return spec


def _execute(channel, name, args, spec, corrections, request_id):
    before = deepcopy(spec)
    before.pop("command", None)
    spec.pop("command", None)
    effects = []
    undo = {"kind": "view", "spec": before}
    data = None
    if name == "save_image":
        return spec, [{"type": "save_image"}], None, {"kind": "none"}, "Saving the current view."
    if name == "undo":
        return _undo(channel, args["command_id"], spec)
    if name == "change_settings":
        return _change_settings(channel, args, spec)
    if name == "run_batch_correction":
        return _correct(channel, spec, request_id)
    if name == "train_reference_model":
        from . import manual_reference, jobs
        project = registry.session_project(channel) or {}
        table, _, _ = manual_reference.context(channel)
        column = match(args["reference_column"], dataio.columns(table), "column", corrections=corrections)
        job_settings = manual_reference.training_settings(channel, column, args["checked_only"])
        job = jobs.start(project["root"], name, job_settings, request_id=request_id)
        return spec, [], {"job":job}, {"kind":"none"}, "Training reference models in the background. Your manual labels stay unchanged."
    if name in {"run_leiden", "run_som", "run_vortex", "name_clusters"}:
        # These jobs work on the combined table, independently of which image/layer is open.
        from .phenotyping_inputs import combined_table
        return _leiden(channel, name, args, spec, combined_table(channel), request_id)
    if name in {"run_consensus_ml", "run_subphenotyping"}:
        from .phenotyping_inputs import combined_table
        table = combined_table(channel)
        return (_consensus(channel, args, spec, table, request_id) if name == "run_consensus_ml"
                else _subphenotyping(channel, args, spec, table, request_id))
    if name == "run_resolve_all":
        from . import jobs, resolve_setup
        from .phenotyping_inputs import combined_table
        project = registry.session_project(channel) or {}
        if not project.get("root"):
            raise ValueError("Load a project before running RESOLVE.")
        values = resolve_setup.job_settings(project["root"], combined_table(channel), args)
        job = jobs.start(project["root"], "resolve_all", values, request_id=request_id)
        return spec, [], {"job": job}, {"kind": "none"}, "Started RESOLVE on the combined table. You can keep using the viewer."
    if name == "tag_cells":
        return _tag_cells(channel, args, spec, request_id)
    if name == "register_pairs" and args.get("scope") != "image":
        return _register(channel, args, spec, None, request_id)
    if name == "compare_pair":
        return _compare_pair(channel, args, undo)
    if name == "open_background_image":
        return _open_background_image(channel, args, undo)
    if name == "segment_images" and args.get("scope") != "image":
        return _segment(channel, args, spec, None, request_id)  # the ticked or all images: no open image needed
    if name == "open_segmentation_image":
        return _open_segmentation_image(channel, args, undo)
    if name == "open_roi_image":
        return _open_roi_image(channel, args, undo)
    if name == "open_annotation_image":
        return _open_annotation_image(channel, args, undo)
    if name == "marker_regions" and args.get("images"):
        return _marker_regions(channel, args, spec, None, request_id)
    if name in {"detect_tissue", "clean_tissue", "detect_artifacts", "subtract_artifacts", "apply_pieces", "make_zones"} and args.get("images"):
        return _tissue(channel, args, spec, None, request_id, kind=name)  # the ticked images: no open image needed
    if name == "detect_roi_cores":
        return _detect_roi_cores(channel, args, spec, undo, request_id)
    if name == "extract_roi_images":
        return _extract_roi_images(channel, args, spec, request_id)
    if name == "export_background":
        return _export_background(channel, args, spec, request_id)
    if name == "detect_background_levels":
        return _detect_background(channel, args, spec, request_id)
    spec, image, info = _view(channel, spec, args, corrections)
    if name == "open_image":
        return spec, [], None, undo, "Opened " + spec["title"] + "."
    if name in {"show_markers", "apply_marker_set"}:
        names = args.get("markers")
        styles = {}
        if name == "apply_marker_set":
            registered = registry.resolve_object(image.get("display_token", ""), "display")
            sets = display.load_sets(display.sets_path(registered["path"])) if registered else {}
            chosen = match(args["set"], list(sets), "marker set", corrections=corrections)
            names = [item["name"] for item in sets[chosen]]
            styles = {item["name"]: item for item in sets[chosen]}
        chosen = [match(value, info.channel_names, "marker", corrections=corrections) for value in names]
        current = {c["index"]: c for c in image.get("channels", [])}
        image["channels"] = [
            {
                **current.get(i, {"index": i, "color": "#4f7dff"}),
                **styles.get(marker, {}),
                "index": i,
                "name": marker,
                "visible": marker in chosen
                or not args.get("replace", True)
                and current.get(i, {}).get("visible", False),
            }
            for i, marker in enumerate(info.channel_names)
        ]
        effects = [{"type": "channels", "layer": image["id"], "channels": image["channels"]}]
        return spec, effects, None, undo, "Showing " + ", ".join(chosen) + "."
    if name in {"save_background_levels", "subtract_background"}:
        return _background(channel, name, args, spec, image, info, request_id)
    if name in {"draw_rois", "extract_rois"}:
        return _rois(channel, name, args, spec, image, request_id, undo)
    if name == "segment_images":
        return _segment(channel, args, spec, image, request_id)
    if name == "register_pairs":
        return _register(channel, args, spec, image, request_id)
    if name == "show_segmentation_region":
        return _show_region(channel, spec, image, undo)
    if name in {"detect_tissue", "clean_tissue", "detect_artifacts", "subtract_artifacts"}:
        return _tissue(channel, args, spec, image, request_id, kind=name)
    if name == "export_classes":
        return _classes(channel, spec, image, request_id)
    if name == "train_cellpose":
        return _train_cellpose(channel, args, spec, request_id)
    if name in {"apply_pieces", "make_zones"}:
        return _tissue(channel, args, spec, image, request_id, kind=name)
    if name == "marker_regions":
        return _marker_regions(channel, args, spec, image, request_id)
    layer, registered, table = _cell_table(spec, image)
    if name in {"select_cells", "refine_selection", "selection_heatmap", "assign_selection", "save_rule", "apply_rule"}:
        from . import selection

        project = registry.session_project(channel) or {}
        available = dataio.columns(table)
        markers = [marker for marker in info.channel_names if marker in available]
        settings = {"table": table, "roi": registered["roi"], "markers": markers, **_keys(registered)}
        token = (spec.get("selection") or {}).get("id")
        if name in {"select_cells", "apply_rule"}:
            if name == "apply_rule":
                saved = selection.library(project["root"])
                selected = match(args["rule_name"], [item["name"] for item in saved], "rule", corrections=corrections)
                rule = {**next(item["rule"] for item in saved if item["name"] == selected), "scope": args["scope"]}
            else:
                rule = deepcopy(args["rule"])
            for condition in rule.get("conditions", []):
                if isinstance(condition, dict):
                    condition["marker"] = match(condition.get("marker", ""), markers, "marker", corrections=corrections)
            data = selection.evaluate(channel, settings, rule)
        else:
            item = selection.get(token, channel=channel, table=table)
            if name == "refine_selection":
                ids = args.get("cells")
                if args.get("points"):
                    if ids:
                        raise ValueError("Choose cells or a drawn region, not both.")
                    import numpy as np
                    from matplotlib.path import Path as PolygonPath

                    found, x = cells.roi_values(table, registered["roi"], "x_centroid", **_keys(registered))
                    other, y = cells.roi_values(table, registered["roi"], "y_centroid", **_keys(registered))
                    if not np.array_equal(found, other):
                        raise ValueError("Cell positions do not match their IDs.")
                    ids = found[
                        PolygonPath(args["points"] + [args["points"][0]]).contains_points(
                            np.column_stack([x, y]), radius=1e-9
                        )
                    ].tolist()
                if ids is None:
                    raise ValueError("Click a cell or draw a region to refine the selection.")
                data = selection.refine(token, channel, registered["roi"], ids, args["mode"])
            elif name == "selection_heatmap":
                return (
                    spec,
                    [],
                    {"heatmap": selection.heatmap(token, channel)},
                    {"kind": "none"},
                    "Marker comparison for the selected cells.",
                )
            elif name == "save_rule":
                selection.save_rule(project["root"], args["rule_name"], item["rule"], markers)
                return spec, [], None, {"kind": "none"}, "Saved rule " + args["rule_name"] + "."
            else:
                count = sum(len(group["ids"]) for group in item["groups"])
                if not count:
                    raise ValueError("No cells are selected.")
                from spacemap import labels

                with labels.write_lock(table):
                    item = selection.get(token, channel=channel, table=table)
                    result = reference.save_groups(
                        table, item["groups"], args["column"], args["phenotype"], request_id, **_keys(registered)
                    )
                undo = {
                    "kind": "selection_labels",
                    "table": table,
                    "column": args["column"],
                    "previous": result["previous"],
                    "expected": args["phenotype"],
                    "keys": _keys(registered),
                    "fingerprint": fingerprint(table),
                }
                layer.update(color_by=args["column"], visible=True, mode="both")
                spec["reference_column"] = args["column"]
                try:
                    summary = selection.reference_summary(item["settings"], args["column"])
                except (ValueError, OSError):
                    summary = {
                        "column": args["column"],
                        "rows": [],
                        "note": "Labels were saved; reopen the reference summary to refresh its counts.",
                    }
                return (
                    spec,
                    [{"type": "labels", "layer": layer["id"], "column": args["column"]}],
                    {"count": count, "column": args["column"], "reference_summary": summary},
                    undo,
                    f"Assigned {count:,} cells to {args['phenotype']}.",
                )
        spec["selection"] = data
        selected_layer = {key: deepcopy(value) for key, value in layer.items() if key not in {"resolve", "command"}}
        selected_layer.update(
            id="studio-selection",
            name="Selected cells",
            selection=data["id"],
            color_by="",
            visible=True,
            mode="fill",
            outline="#00d9ff",
            opacity=0.7,
            added=True,
        )
        spec["layers"] = [item for item in spec["layers"] if item["id"] != "studio-selection"] + [selected_layer]
        return (
            spec,
            [{"type": "selection"}],
            {"selection": data},
            undo,
            f"{data['count']:,} of {data['total']:,} cells selected.",
        )
    if name == "run_neighbourhoods":
        return _leiden(channel, name, args, spec, table, request_id)
    if name == "run_cell_interactions":
        return _interactions(channel, args, spec, table, request_id)
    if name == "run_mr_ml":
        from . import jobs, settings as shared_settings, tasks

        project = registry.session_project(channel) or {}
        if not project.get("root"):
            raise ValueError("Load a project before running a job.")
        from .phenotyping_inputs import combined_table, column_roles, segmentation_column
        table = combined_table(channel)
        roles = column_roles(project["root"], table)
        if args.get('settings_revision'):
            from . import mr_ml
            chosen = mr_ml.values(project['root'])
            if args['settings_revision'] != shared_settings.revision(chosen):
                raise ValueError('MR-ML settings changed. Reload before running.')
            job_settings = mr_ml.job_settings(project['root'], table, roles, chosen, args.get('overwrite', False))
            job = jobs.start(project['root'], 'mr_ml', job_settings, request_id=request_id)
            return spec, [], {'job': job}, {'kind': 'none'}, 'Started MR-ML on the combined table. Results will appear in Jobs.'
        if not args.get('column') or not args.get('reference_column'):
            raise ValueError('Save the MR-ML settings before running, or provide reference and output columns.')
        registered = {**registered, "roi_column": roles["roi"], "label_column": segmentation_column(project["root"], table)}
        available = dataio.columns(table)
        resolve = registry.resolve_object(layer.get("resolve", ""), "resolve") or {}
        options = shared_settings.resolve_values(project["root"])
        markers = [marker for marker in info.channel_names if marker in available]
        job_settings = {
            **resolve,
            "table": table,
            "markers": markers,
            "output_col": args["column"],
            "overwrite_columns": [args["column"]] if args.get("overwrite") else [],
            "cell_id_col": roles["cell_id"] or options.get("cell_id_col") or "cell_id",
            **_keys(registered),
            "options": options,
        }
        cached_settings = shared_settings._resolve_path(project["root"])
        if cached_settings.exists():
            job_settings["settings"] = str(cached_settings)
        if name == "run_mr_ml":
            job_settings["reference_column"] = match(
                args["reference_column"], available, "column", corrections=corrections
            )
            job_settings["methods"] = [
                match(value, available, "column", corrections=corrections) for value in args.get("methods", [])
            ]
            for key in ("confidence_column", "refined_column", "confidence_threshold", "low_confidence_label", "excluded_phenotypes", "morphology_cols"):
                if key in args:
                    job_settings[key] = args[key]
            job_settings["auxiliary_columns"] = [job_settings.get("confidence_column", args["column"] + "_confidence"),
                                                  job_settings.get("refined_column", args["column"] + "_refined")]
            if args.get("overwrite"):
                job_settings["overwrite_columns"] += job_settings["auxiliary_columns"]
            tasks.validate_mr_ml(job_settings)
        tasks.validate_table(job_settings)
        job_settings["revision"] = tasks.source_revision(job_settings)
        job = jobs.start(
            project["root"],
            "mr_ml",
            job_settings,
            request_id=request_id,
        )
        return spec, [], {"job": job}, {"kind": "none"}, "Started the background job. You can keep using the image."
    if name == "show_job_result":
        from . import jobs

        project = registry.session_project(channel) or {}
        job = jobs.status(project.get("root", ""), args["job_id"])
        if job["status"] != "done":
            raise ValueError("This job has not finished successfully yet.")
        result = job.get("result") or {}
        if result.get("table") != table:
            raise ValueError("Open an image using this job’s cell table to see its result.")
        if result.get("column") not in dataio.columns(table):
            raise ValueError("The saved result column is no longer available.")
        added = {key: deepcopy(value) for key, value in layer.items() if key not in {"resolve", "command"}}
        added.update(
            id="result-" + args["job_id"],
            name=result["column"],
            color_by=result["column"],
            visible=True,
            mode="both",
            added=True,
        )
        if any(item["id"] == added["id"] for item in spec["layers"]):
            raise ValueError("This result is already in the layer list.")
        spec["layers"].append(added)
        return spec, [{"type": "add_layer", "layer": added}], result, undo, "Showing " + result["column"] + "."
    if name == "density" and args.get("table"):
        return _point_density(channel, spec, layer, registered, image, args, corrections, request_id, undo)
    if name in {"colour_by", "filter_cells", "density", "count_cells"}:
        column = _column(table, layer, args.get("column"), corrections)
        if name == "count_cells" and args["scope"] == "project":
            values = dataio.load(table, columns=[column])[column]
            groups = values.fillna("Unassigned").astype(str).replace({"": "Unassigned"}).value_counts().to_dict()
        else:
            groups = _groups(table, registered, column)
        chosen = []
        for value in args.get("phenotypes", []) or ([args["phenotype"]] if args.get("phenotype") else []):
            chosen.append(match(value, groups, "phenotype", corrections=corrections))
        if name == "count_cells":
            if chosen:
                groups = {value: groups.get(value, 0) for value in chosen}
            data = {
                "column": column,
                "scope": args["scope"],
                "counts": {str(k): int(v) for k, v in groups.items()},
                "total": int(sum(groups.values())),
            }
            return spec, [], data, {"kind": "none"}, f"{data['total']:,} cells counted."
        if name == "density":
            density = {
                "id": "density-" + request_id,
                "type": "density",
                "token": layer["token"],
                "image_token": image["token"],
                "name": chosen[0] + " density",
                "visible": True,
                "opacity": 0.65,
                "table": layer["table"],
                "tables": layer.get("tables", []),
                "fcol": column,
                "fvals": chosen,
                "radius": args["radius"],
                "cmap": "magma",
                "open": True,
            }
            spec["layers"].append(density)
            effects = [{"type": "add_layer", "layer": density}]
        else:
            layer.update(color_by=column, visible=True, mode="both")
            if chosen:
                layer["hidden"] = [value for value in groups if value not in chosen]
            else:
                layer["hidden"] = []
            effects = [
                {
                    "type": "cells",
                    "layer": layer["id"],
                    "changes": {key: layer[key] for key in ("color_by", "visible", "mode", "hidden")},
                }
            ]
        return (
            spec,
            effects,
            None,
            undo,
            ("Showing " + ", ".join(chosen) + ".") if chosen else "Cells coloured by " + column + ".",
        )
    if name == "zoom_to_cell":
        detail = cells.cell_details(table, registered["roi"], args["cell"], **_keys(registered))
        x, y = detail.get("x_centroid"), detail.get("y_centroid")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not math.isfinite(x + y):
            raise ValueError("That cell has no saved position in this image.")
        spec["bounds"] = [x - 100, y - 100, 200, 200]
        return (
            spec,
            [{"type": "zoom_cell", "layer": layer["id"], "cell": args["cell"], "x": x, "y": y}],
            None,
            undo,
            "Showing cell " + str(args["cell"]) + ".",
        )
    if name == "assign_phenotype":
        ids = sorted(set(args["cells"]))
        found, _ = cells.assignment_rows(table, registered["roi"], ids, **_keys(registered))
        if sorted(found.tolist()) != ids:
            raise ValueError("Some cell numbers do not belong to this ROI. Nothing was assigned.")
        result = reference.save(
            table,
            registered["roi"],
            args["column"],
            dict.fromkeys(ids, args["phenotype"]),
            request_id,
            **_keys(registered),
        )
        undo = {
            "kind": "labels",
            "table": table,
            "roi": registered["roi"],
            "column": args["column"],
            "previous": result["previous"],
            "expected": dict.fromkeys(ids, args["phenotype"]),
            "keys": _keys(registered),
            "fingerprint": fingerprint(table),
        }
        layer.update(color_by=args["column"], visible=True, mode="both")
        effects = [{"type": "labels", "layer": layer["id"], "column": args["column"]}]
        return (
            spec,
            effects,
            {"count": len(ids), "column": args["column"]},
            undo,
            f"Assigned {len(ids):,} cells to {args['phenotype']}.",
        )
    if name in {"set_threshold", "save_thresholds"}:
        resolve = registry.resolve_object(layer.get("resolve", ""), "resolve")
        if not resolve:
            raise ValueError("Open Live RESOLVE for this image first.")
        session = live_resolve.open_session(resolve)
        if name == "set_threshold":
            marker = match(args["marker"], session.markers, "marker", corrections=corrections)
            previous = session.thresholds[marker]
            live_resolve.assign(session, {marker: args["value"]})
            undo = {
                "kind": "threshold",
                "resolve": resolve,
                "marker": marker,
                "previous": previous,
                "expected": args["value"],
            }
            effects = [{"type": "threshold", "layer": layer["id"], "marker": marker, "value": args["value"]}]
            return spec, effects, None, undo, f"{marker} threshold set to {args['value']:g} raw units."
        path = Path(resolve["thresholds"])
        old = path.read_text() if path.exists() else None
        if old is not None and len(old) > 16 * 1024 * 1024:
            raise ValueError("This threshold file is too large for safe Undo.")
        live_resolve.save(session)
        undo = {"kind": "file", "path": str(path), "previous": old, "expected": fingerprint(path)}
        return spec, [], None, undo, "Saved thresholds for " + registered["roi"] + "."
    raise ValueError("This command is not available yet.")


def _background(channel, name, args, spec, image, info, request_id):
    """Tab 2 through Studio: save this image's levels, or remove background as a job."""
    from spacemap_gui.studio.settings import read_config

    from . import background, jobs

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    images, tables, outputs = background.folders(root, mappings, config)
    path = Path(registry.resolve(image["token"]))
    if name == "save_background_levels":
        levels = args["levels"]
        unknown = sorted(set(levels) - set(info.channel_names))
        if unknown:
            raise ValueError("These markers are not in this image: " + ", ".join(unknown[:8]) + ".")
        target = background.table_path(tables, path.name)
        old = target.read_text() if target.exists() else None
        if old is not None and len(old) > 16 * 1024 * 1024:
            raise ValueError("This threshold table is too large for safe Undo.")
        background.save_levels(target, levels, args.get("method"))
        undo = {"kind": "file", "path": str(target), "previous": old, "expected": fingerprint(target)}
        data = {"table": target.name, "markers": len(levels)}
        return (
            spec,
            [],
            data,
            undo,
            f"Saved {len(levels)} background levels for {background.stem(path.name)} in {target.parent.name}.",
        )
    if args["scope"] == "image":
        chosen = [path]
    else:
        chosen = sorted(
            candidate
            for candidate in images.glob("*")
            if candidate.name.lower().endswith(background.IMAGE_SUFFIXES)
            and background.table_path(tables, candidate.name).is_file()
        )
        if not chosen:
            raise ValueError("No image in " + images.name + " has saved background levels yet.")
    missing = [item.name for item in chosen if not background.table_path(tables, item.name).is_file()]
    if missing:
        raise ValueError("Save background levels for " + ", ".join(missing[:6]) + " first.")
    settings = {
        "images": [
            {
                "image": str(item),
                "table": str(background.table_path(tables, item.name)),
                "output": str(background.output_path(outputs, item.name)),
            }
            for item in chosen
        ]
    }
    job = jobs.start(str(root), "subtract_background", settings, request_id=request_id)
    count = len(chosen)
    return (
        spec,
        [],
        {"job": job},
        {"kind": "none"},
        f"Removing background from {count} image{'s' if count > 1 else ''} in the background. The originals stay.",
    )


def _rois(channel, name, args, spec, image, request_id, undo):
    """Tab 3 through Studio: draw ROIs into its file, or cut them out as a job."""
    from spacemap_gui.studio.settings import read_config
    from spacemap_gui.viewer import ui as viewer_ui

    from . import jobs, rois

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    images, shapes_dir, outputs = rois.folders(root, mappings, config)
    picks = rois._saved(config, rois.FILES_KEY, {}) or {}
    path = Path(registry.resolve(image["token"]))
    if name == "draw_rois":
        target = rois.shapes_file(shapes_dir, path.name, picks.get(path.name))
        if any(layer.get("id") == "rois" for layer in spec["layers"]):
            return spec, [], {"file": target.name}, {"kind": "none"}, "The ROI layer is already open."
        sources = (
            [(f"{p.parent.name} / {p.name}", str(p)) for p in sorted(shapes_dir.glob("*.geojson")) if p != target][:60]
            if shapes_dir.is_dir()
            else []
        )
        layer = viewer_ui.shapes_layer(
            target,
            str(path),
            name="ROIs",
            layer_id="rois",
            sources=sources,
            classes=[{"name": "ROI", "color": "#ffd600"}],
            default_class="ROI",
        )
        layer["added"] = True
        spec["layers"].append(layer)
        count = len(rois.roi_names(target))
        return (
            spec,
            [{"type": "add_layer", "layer": layer}],
            {"file": target.name, "count": count},
            undo,
            f"Draw and name ROIs on the image; they are saved into {target.name} ({count} so far).",
        )
    chosen = (
        [path]
        if args["scope"] == "image"
        else sorted(
            candidate
            for candidate in images.glob("*")
            if candidate.name.lower().endswith(rois.IMAGE_SUFFIXES)
            and rois.roi_names(rois.shapes_file(shapes_dir, candidate.name, picks.get(candidate.name)))
        )
    )
    missing = [item.name for item in chosen if not rois.roi_names(rois.shapes_file(shapes_dir, item.name, picks.get(item.name)))]
    if missing or not chosen:
        raise ValueError("Draw ROIs on " + (", ".join(missing[:6]) if missing else "an image") + " first.")
    from spacemap.viewer.display import settings_path
    settings = {
        "images": [
            {
                "image": str(item),
                "rois": str(rois.shapes_file(shapes_dir, item.name, picks.get(item.name))),
                "outputs": str(outputs),
                "display": str(settings_path(root, item)),
                "mpp": config.get("mpp_codex"),
                "pair": rois.pair_settings(config, item.name),
                "keep_outside": bool(args.get("keep_outside")),
            }
            for item in chosen
        ]
    }
    job = jobs.start(str(root), "extract_rois", settings, request_id=request_id)
    count = len(chosen)
    return (
        spec,
        [],
        {"job": job},
        {"kind": "none"},
        f"Cutting out the ROIs of {count} image{'s' if count > 1 else ''} in the background; ROIs already extracted are kept.",
    )


def _regions_layer(config, places, path, sample):
    """The image's marker regions, to review, name and edit (saved into its regions file as they change)."""
    from spacemap_gui.viewer import ui as viewer_ui

    from . import annotation, pixelclasses

    try:
        names = pixelclasses.common_names()
    except Exception:  # suggestions only
        names = []
    layer = viewer_ui.shapes_layer(annotation.marker_regions(places["output"], sample), str(path), name="Marker regions", layer_id="marker-regions",
                                   classes=[{"name": "Region", "color": "#ffd600"}], default_class="Region", regions=True, names=names,
                                   reference=annotation.valid_tissue(places["output"], sample), annotation_kind="regions")
    layer["added"] = True
    return layer


def _marker_regions(channel, args, spec, image, request_id):
    """Marker regions: the open image at once (its regions layer read again), the ticked images as a job."""
    from . import annotation, annotation_edits, annotation_inputs, jobs, regions

    root, config, mappings = _background_project(channel)
    places = annotation.all_folders(root, mappings)
    values = regions.settings(config)
    if not values["rules"]:
        raise ValueError("Add at least one marker and its level first.")
    if args.get("images"):
        missing = [name for name in args["images"] if "/" in name or not (places["images"] / name).is_file()]
        if missing:
            raise ValueError("Not in the images folder: " + ", ".join(missing[:4]))
        paths = [places["images"] / name for name in dict.fromkeys(args["images"])]
    else:
        paths = [Path(registry.resolve(image["token"]))]
    items = [{"root":str(root), "image": str(path), "values": {**values, "within":"tissue"},
              "selection":annotation_inputs.selection(root,path,'regions'),
              "target": str(regions.target(places['output'],path,values,annotation_inputs.selection(root,path,'regions'))),
              "tissue": str(annotation_inputs.scope_path(root, path, places["output"], "regions"))} for path in paths]
    for item in items:
        item["revision"] = annotation_edits.revision(item["target"])
        if values["cutoff"] == "saved":
            item["levels"] = regions.saved_levels(root, config, mappings, item["image"])  # this image's own levels
        if values["cutoff"] == "groups":
            item["pool"] = regions.pool_for(root, mappings, item["image"], values)  # the image, or its slide's ROIs
    if args.get("images"):
        job = jobs.start(str(root), "marker_regions", {"images": items}, request_id=request_id)
        return (spec, [], {"job": job}, {"kind": "none"},
                f"Finding marker regions in {len(items)} image{'s' if len(items) > 1 else ''} in the background; each image\'s regions file is replaced.")
    item = items[0]
    features, stats = regions.find(item["image"], item['values'], item["tissue"], levels=item.get("levels"), pool=item.get("pool"))
    regions.write(item["target"], features, item["revision"])
    regions.register_result(item)
    from . import annotation_layers
    layer = annotation_layers.layer(root,paths[0],places['output'],item['target'])
    spec["layers"] = [entry for entry in spec["layers"] if entry.get("id") != layer['id']] + [layer]
    return (spec, [{"type": "reload_shapes", "layer": layer['id']}], {"regions": stats["regions"], "layer":layer}, {"kind": "none"},
            f"{stats['regions']} marker region{'s' if stats['regions'] != 1 else ''} in {paths[0].name} ({stats['area_um2']:,.0f} µm², read at "
            f"{stats['level_microns']} µm/px; cutoffs: {regions.describe(stats['cutoffs'])}). Review and name them; they are saved as you change them.")


def _open_annotation_image(channel, args, undo):
    """An image of the Annotation step with its valid tissue and the artifacts drawn by hand (to draw more)."""
    from spacemap_gui.viewer import ui as viewer_ui

    from . import annotation

    root, config, mappings = _background_project(channel)
    plan = annotation.images_plan(root, mappings, config)
    row = next((row for row in plan["rows"] if row["image"] == args["image"]), None)
    if row is None:
        raise ValueError(f"{args['image']} is not in the images folder.")
    places = annotation.all_folders(root, mappings)
    path = (places["images"] / row["image"]).resolve()
    if not (registry.session_project(channel) or {}).get("read_only"):
        annotation.unify_tissue(places["output"], row["sample"], label=annotation.tissue_label(config))
    viewer_ui.window_images(channel)
    entries = registry.session_images(channel) or []
    image_id = next((i for i, entry in enumerate(entries) if Path(entry["path"]).resolve() == path), None)
    if image_id is None:
        raise ValueError(f"{row['image']} could not be listed for the viewer.")
    spec = viewer_ui.window_image_spec(channel, image_id)
    added = _annotation_layers(config, places, path, row["sample"], root)
    ids = {layer["id"] for layer in added}
    editable_paths = {layer.get('saved_file') for layer in added}
    def superseded(layer):
        original = registry.resolve_object(layer.get('token', ''), 'annotations') or {}
        return layer.get('id') in ids or (original.get('path') and str(Path(original['path']).resolve()) in editable_paths)
    spec["layers"] = [layer for layer in spec["layers"] if not superseded(layer)] + added
    done = [stage for stage in annotation.STAGES if row["done"][stage]]
    return (spec, [], {"image": row["image"], "sample": row["sample"]}, undo,
            f"{row['image']}" + (f": {', '.join(done)} done." if done else ": nothing done yet."))


def _annotation_layers(config, places, path, sample, root=None):
    from spacemap_gui.viewer import ui as viewer_ui
    from . import annotation

    added = []
    tissue = annotation.valid_tissue(places["output"], sample)
    layer = viewer_ui.shapes_layer(tissue, str(path), name=annotation.tissue_label(config), layer_id="tissue-regions", regions=True,
                                   classes=[{"name": "Valid_Tissue", "color": "#2ecc71"}], default_class="Valid_Tissue",
                                   annotation_kind="tissue", tissue_label=annotation.tissue_label(config), reference=tissue)
    layer["added"] = True
    layer["working_tissue"] = True
    added.append(layer)
    from spacemap_gui.viewer.artifact_api import _layer
    from . import artifact_learning
    for part in artifact_learning.NAMES:
        if artifact_learning.polygons_path(places["output"], sample, part).is_file():
            reviewed = _layer(places["output"], sample, part, str(path))
            reviewed.update(added=True, visible=False)
            added.append(reviewed)
    drawn = viewer_ui.shapes_layer(annotation.drawn_artifacts(places["output"], sample), str(path), name="Artifacts",
                                   layer_id="drawn-artifacts", classes=[{"name": annotation.DRAWN_CLASS, "color": "#ff5252"}],
                                   default_class=annotation.DRAWN_CLASS, regions=True, annotation_kind="artifacts", reference=tissue)
    drawn["added"] = True
    added.append(drawn)
    from . import annotation_inputs
    if root is not None:
        added.extend(annotation_inputs.layer(target, path) for target in annotation_inputs.imported(root, path))
    added.append(_regions_layer(config, places, path, sample))
    if root is not None:
        from . import annotation_layers
        added.extend(annotation_layers.layer(root, path, places['output'], target)
                     for target in annotation_layers.generated(root, path, places['output'])
                     if Path(target) != annotation.marker_regions(places['output'], sample).resolve())
    return added


def _roi_step(channel):
    """The ROI step's project, folders, list and the ROI file of each image."""
    from . import rois

    root, config, mappings = _background_project(channel)
    plan = rois.images_plan(root, mappings, config)
    places = {kind: Path(folder) for kind, folder in plan["folders"].items()}
    return root, config, plan, places, {row["image"]: row for row in plan["rows"]}


def _roi_layer(places, path, file):
    """The image's ROIs as a layer to draw, name and check (saved into its ROI file as they change)."""
    from spacemap_gui.viewer import ui as viewer_ui

    target = places["shapes"] / file
    sources = [(f"{p.parent.name} / {p.name}", str(p)) for p in sorted(places["shapes"].glob("*.geojson")) if p != target][:60] \
        if places["shapes"].is_dir() else []
    layer = viewer_ui.shapes_layer(target, str(path), name="ROIs", layer_id="rois", sources=sources,
                                   classes=[{"name": "ROI", "color": "#ffd600"}], default_class="ROI")
    layer["added"] = True
    return layer


def _open_roi_image(channel, args, undo):
    """An image of the ROI step with its ROI file as a layer."""
    from spacemap_gui.viewer.ui import window_image_spec, window_images

    from . import rois

    _, _, _, places, rows = _roi_step(channel)
    row = rows.get(args["image"])
    if row is None:
        raise ValueError(f"{args['image']} is not in the images folder.")
    path = (places["images"] / row["image"]).resolve()
    window_images(channel)
    entries = registry.session_images(channel) or []
    image_id = next((i for i, entry in enumerate(entries) if Path(entry["path"]).resolve() == path), None)
    if image_id is None:
        raise ValueError(f"{row['image']} could not be listed for the viewer.")
    spec = window_image_spec(channel, image_id)
    layer = _roi_layer(places, path, row["file"])
    spec["layers"] = [item for item in spec["layers"] if item.get("id") != "rois"] + [layer]
    count = len(rois.roi_names(places["shapes"] / row["file"]))
    return (spec, [], {"image": row["image"], "file": row["file"], "count": count}, undo,
            f"{row['image']} with its ROIs ({count}, from {row['file']}). Draw, name or delete them on the image; they save as you go.")


def _open_segmentation_image(channel, args, undo):
    """An image of the Segmentation step, with its cells shown when it is segmented, and the area it is segmented in."""
    from spacemap_gui.studio.settings import read_config
    from spacemap_gui.viewer.ui import window_image_spec, window_images

    from . import segmentation

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    places = segmentation.step_folders(root, config, mappings)
    path = (places["images"] / args["image"]).resolve()
    if not path.is_file() or path.parent != places["images"].resolve():
        raise ValueError(f"{args['image']} is not in the images folder.")
    window_images(channel)
    entries = registry.session_images(channel) or []
    image_id = next((i for i, entry in enumerate(entries) if Path(entry["path"]).resolve() == path), None)
    if image_id is None:
        raise ValueError(f"{args['image']} could not be listed for the viewer.")
    spec = window_image_spec(channel, image_id)
    values = segmentation.values(root, config)
    kind, area = segmentation.area_of(values, path.name)
    area_path = None
    if kind == "layer":
        saved = segmentation.saved_annotations(root, config, mappings, [path.name])[path.name].get(area)
        area_path = saved if isinstance(saved, str) else (saved or {}).get("path")
    cells = None
    for layer in spec["layers"]:
        if layer.get("type") == "cells":
            layer.update(visible=True, added=True)  # this step's own result: shown
            cells = layer
        elif layer.get("type") == "annotations" and area_path:
            found = registry.resolve_object(layer.get("token", ""), "annotations") or {}
            if found.get("path") and Path(found["path"]).resolve() == Path(area_path).resolve():
                layer.update(visible=True, added=True, opacity=0.35)  # where it is segmented
    table, _ = segmentation.results(places["output"], path.name)
    count = segmentation.cell_count(table) if table.is_file() else None
    where = area if kind == "layer" else ("the whole image" if kind == "whole" else "the annotation file's classes")
    if count is None:
        message = f"{path.name}: not segmented yet; it will be segmented in {where}."
    elif cells:
        message = f"{path.name}: {count:,} cells, shown on the image (Cells, in Layers)."
    else:
        message = f"{path.name}: {count:,} cells" + ("; their outlines were not found to show." if count else " were found: try another model or diameter.")
    return (spec, [], {"image": path.name, "cells": count}, undo, message)


def _detect_roi_cores(channel, args, spec, undo, request_id):
    """TMA cores as ROIs: the open image at once (its ROI layer refreshed), the other listed images as a job."""
    from . import jobs, rois

    root, config, plan, places, rows = _roi_step(channel)
    unknown = [name for name in args["images"] if name not in rows]
    if unknown:
        raise ValueError("Not in the images folder: " + ", ".join(unknown[:4]))
    names = list(dict.fromkeys(args["images"]))
    items = [{"image": str(places["images"] / name), "target": str(places["shapes"] / rows[name]["file"]), "settings": plan["tma"]} for name in names]
    effects, parts, data = [], [], {}
    opened = args.get("open") if args.get("open") in names else None
    if opened:
        item = items[names.index(opened)]
        count = rois.detect_cores(item["image"], item["target"], item["settings"])
        layer = _roi_layer(places, (places["images"] / opened).resolve(), rows[opened]["file"])
        spec["layers"] = [entry for entry in spec["layers"] if entry.get("id") != "rois"] + [layer]
        effects.append({"type": "reload_shapes", "layer": "rois"})  # the same file, rewritten: read again
        parts.append(f"{count} core{'s' if count != 1 else ''} found on {opened}" + ("" if count else ": change the core sizes or threshold"))
        data["cores"] = count
    others = [item for item in items if Path(item["image"]).name != opened]
    if others:
        data["job"] = jobs.start(str(root), "detect_roi_cores", {"images": others}, request_id=request_id)
        parts.append(f"finding the cores of {len(others)} other image{'s' if len(others) > 1 else ''} in the background")
    message = "; ".join(parts) + ". They are saved as the ROIs: check them on each image, then Extract."
    return spec, effects, data, {"kind": "none"}, message[0].upper() + message[1:]


def _extract_roi_images(channel, args, spec, request_id):
    """Cut out the ROIs of the listed images, as a job."""
    from . import jobs, rois

    root, config, plan, places, rows = _roi_step(channel)
    unknown = [name for name in args["images"] if name not in rows]
    if unknown:
        raise ValueError("Not in the images folder: " + ", ".join(unknown[:4]))
    names = list(dict.fromkeys(args["images"]))
    empty = [name for name in names if not rows[name]["rois"]]
    names = [name for name in names if rows[name]["rois"]]
    if not names:
        raise ValueError("No ROIs yet on " + ", ".join(empty[:4]) + ": draw them, or pick their GeoJSON file in the list.")
    from spacemap.viewer.display import settings_path
    items = [{
        "image": str(places["images"] / name), "rois": str(places["shapes"] / rows[name]["file"]), "outputs": str(places["output"]),
        "display": str(settings_path(root, places["images"] / name)), "mpp": config.get("mpp_codex"), "pair": rois.pair_settings(config, name),
        "keep_outside": args["keep_outside"],
    } for name in names]
    job = jobs.start(str(root), "extract_rois", {"images": items}, request_id=request_id)
    total = sum(rows[name]["rois"] for name in names)
    return (spec, [], {"job": job}, {"kind": "none"},
            f"Cutting out {total} ROI{'s' if total != 1 else ''} of {len(names)} image{'s' if len(names) > 1 else ''} in the background"
            + (" (whole boxes)" if args["keep_outside"] else " (outside each shape blanked)") + "; ROIs already extracted are kept"
            + (f"; no ROIs yet: {', '.join(empty[:4])}" if empty else "") + ".")


def _segment(channel, args, spec, image, request_id):
    """Tab 5 through Studio, as a job: the open image, the ticked images, or every image in the step's images folder.

    Each image is segmented where its row says (the step's choice, the whole image, or a saved annotation such as its
    tissue, all its polygons together, as it is now), or only inside the polygons picked on the open image. Images
    already segmented are kept unless ``again``: then their results move to ``previous/`` first. ``tag`` tags the cells
    with the saved rules as they are segmented.
    """
    from spacemap_gui.studio.settings import read_config

    from . import jobs, segmentation

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    places = segmentation.step_folders(root, config, mappings)
    inputs, outputs = places["images"], places["output"]
    if args["scope"] == "image":
        if not image:
            raise ValueError("Open an image first.")
        chosen = [Path(registry.resolve(image["token"]))]
    elif args["scope"] == "images":
        names = list(dict.fromkeys(args.get("images") or []))
        if not names:
            raise ValueError("Tick the images first.")
        missing = [name for name in names if not (inputs / name).is_file()]
        if missing:
            raise ValueError("Not in the images folder: " + ", ".join(missing[:4]))
        chosen = [inputs / name for name in names]
    else:
        chosen = sorted(item for item in inputs.glob("*") if item.name.lower().endswith(segmentation.IMAGE_SUFFIXES))
    if not chosen:
        raise ValueError("No images to segment in " + inputs.name + ".")
    values = segmentation.values(root, config)
    # Tagging is the Cell tagging step's (its columns and names), run after segmenting when asked; never classic
    # Tab 5's own rules inside the segmentation.
    values["tab4_spatial_enabled"] = False
    done = [item for item in chosen if segmentation.results(outputs, item.name)[0].is_file()]
    if done and not args.get("again"):
        chosen = [item for item in chosen if item not in done]
        if not chosen:
            return (spec, [], {}, {"kind": "none"},
                    f"{'All ' + str(len(done)) + ' images are' if len(done) > 1 else done[0].name + ' is'} segmented already; "
                    "tick Segment again to redo them (the earlier results are kept in previous/).")
    geojson = segmentation.step_folders(root, config, mappings)["geojson"]
    items = [{"image": str(item), "outputs": str(outputs), "geojson": str(geojson)} for item in chosen]
    if args.get("shape"):
        # Only inside the polygons picked on the open image.
        from .annotation_edits import shown

        layer = registry.resolve_object(args["shape"], "shapes")
        if args["scope"] != "image" or not layer or Path(layer["image"]).resolve() != chosen[0].resolve():
            raise ValueError("Selected polygons belong to the open image: pick them on it, then use This image.")
        wanted = {str(item) for item in args.get("ids") or []}
        picked = [f for f in shown(layer) if str(f["properties"]["id"]) in wanted]
        items[0]["area"] = {"name": f"{len(picked)} selected polygon{'s' if len(picked) != 1 else ''}",
                            "path": str(segmentation.selected_mask(root, chosen[0], picked)), "exact": True}
    else:
        # Each image where its row says: the whole image, a saved annotation (its own), or classic Tab 5's choice.
        areas = {item.name: segmentation.area_of(values, item.name) for item in chosen}
        layered = [name for name, (kind, _) in areas.items() if kind == "layer"]
        found = segmentation.saved_annotations(root, config, mappings, layered) if layered else {}
        missing = [f"{name} ({areas[name][1]})" for name in layered if areas[name][1] not in found[name]]
        if missing:
            raise ValueError("No saved annotation to segment in for " + ", ".join(missing[:4])
                             + (f" and {len(missing) - 4} more" if len(missing) > 4 else "")
                             + ". Save it in the Annotation step, or choose another area in its row.")
        for item in items:
            kind, name = areas[Path(item["image"]).name]
            if kind == "whole":
                item["area"] = {"whole": True}
            elif kind == "layer":  # its polygons are cut out in the job, not while the button waits
                item["area"] = {"name": name, "pending": found[Path(item["image"]).name][name]}
    moved = [segmentation.set_aside(outputs, item.name, geojson) for item in chosen if item in done] if args.get("again") else []
    job = jobs.start(str(root), "segment_images", {"images": items, "values": values, "tag": str(root) if args.get("tag") else None,
                                                   "root": str(root), "mappings": mappings},
                     request_id=request_id)
    count = len(items)
    kept = len(done) if done and not args.get("again") else 0
    return (
        spec,
        [],
        {"job": job, "images": [Path(item["image"]).name for item in items]},
        {"kind": "none"},
        f"Segmenting {count} image{'s' if count > 1 else ''} in the background"
        + (" on the GPU" if values.get("tab4_gpu") else " on the CPU")
        + (f"; {kept} already segmented, kept" if kept else "")
        + (f"; earlier results of {len(moved)} moved to previous/" if moved else "") + ".",
    )


def _register(channel, args, spec, image, request_id):
    """Tab 1 through Studio, as a job: the open image's pair, the listed pairs, or every unregistered pair.

    Pairs are Tab 1's (its naming rule and per-image choices). ``again`` registers listed pairs that
    already have a merged image once more; the earlier image is moved into ``previous/`` first.
    """
    from spacemap.viewer.catalog import sample_of
    from spacemap_gui.studio.settings import read_config

    from . import jobs, registration

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    options = registration.check(args)
    found = registration.pairs(root, mappings, config)
    _, _, registered = registration.folders(root, mappings)
    done = {sample for sample in found if registration.output_for(registered, sample).exists()}
    if args["scope"] == "image":
        name = Path(registry.resolve(image["token"])).name
        codex = {Path(pair["codex"]).name: sample for sample, pair in found.items()}
        he = {Path(pair["he"]).name: sample for sample, pair in found.items()}
        sample = codex.get(name) or he.get(name) or sample_of(name)
        if sample not in found:
            raise ValueError(f"No CODEX and H&E pair found for {name}. Choose its pair in the Pairs list.")
        chosen = [sample]
    elif args["scope"] == "pairs":
        chosen = list(dict.fromkeys(args.get("samples") or []))
        missing = [sample for sample in chosen if sample not in found]
        if not chosen or missing:
            raise ValueError("Choose pairs from the list" + (": no pair " + ", ".join(missing) if missing else "."))
        if not args["again"]:
            chosen = [sample for sample in chosen if sample not in done]
            if not chosen:
                raise ValueError("These pairs are already registered. Use Register again to redo them.")
    else:
        chosen = [sample for sample in sorted(found) if sample not in done]
        if not chosen:
            raise ValueError("Every pair is already registered.")
    again = args["again"] and args["scope"] != "all"
    settings = {
        "pairs": [
            {
                "sample": sample,
                **found[sample],
                "output": str(registration.output_for(registered, sample)),
                "again": again and sample in done,
            }
            for sample in chosen
        ],
        "options": options,
    }
    job = jobs.start(str(root), "register_pairs", settings, request_id=request_id)
    count = len(chosen)
    redo = sum(item["again"] for item in settings["pairs"])
    return (
        spec,
        [],
        {"job": job},
        {"kind": "none"},
        f"Registering {count} pair{'s' if count > 1 else ''} in the background"
        + (f"; {redo} earlier result{'s' if redo > 1 else ''} will be kept in previous/." if redo else "; registered pairs are kept."),
    )


def _tissue(channel, args, spec, image, request_id, kind="detect_tissue"):
    """Tab 4 through Studio as a job: tissue (step 1), artifacts (2), pieces (2b) or zones (4)."""
    from spacemap_gui.studio.settings import read_config

    from . import annotation, annotation_edits, artifact_learning, jobs

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    images, output = annotation.folders(root, mappings)
    if args.get("images"):
        missing = [name for name in args["images"] if "/" in name or not (images / name).is_file()]
        if missing:
            raise ValueError("Not in the images folder: " + ", ".join(missing[:4]))
        chosen = [images / name for name in dict.fromkeys(args["images"])]
    else:
        chosen = (
            [Path(registry.resolve(image["token"]))]
            if args["scope"] == "image"
            else sorted(item for item in images.glob("*") if item.name.lower().endswith(annotation.IMAGE_EXTENSIONS))
        )
    if not chosen:
        raise ValueError("No images in " + images.name + ".")
    zones = annotation.zones_folder(root, mappings)
    settings = {
        "images": [
            {"image": str(item), "sample": annotation.sample_id(item.name), "output": str(output), "zones": str(zones),
             "tissue_revision": annotation_edits.revision(annotation.valid_tissue(output, annotation.sample_id(item.name))),
             "cleaned_revision": annotation_edits.revision(annotation.cleaned_tissue(output, annotation.sample_id(item.name))),
             "reviewed_artifact_revisions": {part: annotation_edits.revision(artifact_learning.polygons_path(output, annotation.sample_id(item.name), part)) for part in artifact_learning.NAMES},
             "artifact_revision": annotation_edits.revision(annotation.drawn_artifacts(output, annotation.sample_id(item.name)))}
            for item in chosen
        ],
        "config": {
            "tab_configurations": {
                "tab4_annotation": (config.get("tab_configurations") or {}).get("tab4_annotation") or {}
            }
        },
        "values": annotation.settings(config),
    }
    if kind in ('apply_pieces', 'subtract_artifacts', 'clean_tissue'):
        from . import annotation_inputs
        for item in settings["images"]:
            annotation_inputs.prepare_working(root, config, item['image'], output)
            target = annotation_inputs.working_path(root, item['image'], output)
            if not target.is_file(): raise ValueError("Save the selected Tissue annotation first.")
            item["tissue_path"] = str(target)
            item["tissue_revision"] = annotation_edits.revision(target)
            item['class_name'] = annotation_inputs.selection(root,item['image'],'tissue').get('class_name','')
            item['tissue_kind'] = 'tissue' if target == annotation.valid_tissue(output,item['sample']) else 'regions'
    settings['root'] = str(root)
    job = jobs.start(str(root), kind, settings, request_id=request_id)
    if image and kind != "make_zones":
        path = Path(registry.resolve(image["token"]))
        layers = _annotation_layers(config, {"output": output}, path, annotation.sample_id(path.name), root)
        ids = {layer["id"] for layer in spec["layers"]}
        spec["layers"].extend(layer for layer in layers if layer["id"] not in ids)
    count = len(chosen)
    doing = {
        "detect_tissue": "Detecting tissue in",
        "clean_tissue": "Removing artifacts from the tissue of",
        "detect_artifacts": "Detecting artifacts in",
        "subtract_artifacts": "Subtracting reviewed artifacts from",
        "apply_pieces": "Splitting the tissue into pieces for",
        "make_zones": "Building the zones of",
    }[kind]
    return (
        spec,
        [],
        {"job": job},
        {"kind": "none"},
        f"{doing} {count} image{'s' if count > 1 else ''} in the background; Tab 4's annotation files are written.",
    )


def _check_settings_revision(args, values):
    from .settings import revision
    if args.get("settings_revision") and args["settings_revision"] != revision(values):
        raise ValueError("The settings changed before the run started. Reload the panel and check your choices.")


def _leiden(channel, name, args, spec, table, request_id):
    """Tab 7's Leiden or Tab 8's neighbourhoods through Studio: run one, or name its groups, each as a job."""
    from spacemap_gui.studio.settings import read_config

    from . import jobs, leiden, tasks

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project before running a job.")
    root = project["root"]
    if name == "run_leiden":
        values = leiden.values(root)
        _check_settings_revision(args, values)
        from .phenotyping_inputs import column_roles
        roles = column_roles(root, table)
        values = {**values, "cell_id_col": roles["cell_id"] or values["cell_id_col"]}
        if not values["markers"]:
            raise ValueError("Choose the markers for Leiden and save the settings first.")
        bc = ((read_config(root).get("tab_configurations") or {}).get("tab6_batch_correction") or {}).get("bc_config")
        cofactor = float((bc if isinstance(bc, dict) else {}).get("arcsinh_cofactor", 5.0) or 5.0)
        job_settings = {
            "table": table,
            "markers": list(values["markers"]),
            "output_col": values["output_col"] + "_clusters",
            "phenotype_col": values["output_col"],
            "cell_id_col": values["cell_id_col"] or "cell_id",
            "roi_col": roles["roi"],
            "project_root": str(root),
            "pipeline": leiden.pipeline_settings(values, table, cofactor),
            "overwrite_columns": [values["output_col"], values["output_col"] + "_clusters"] if args.get("overwrite") else [],
        }
        tasks.validate_table(job_settings)
        if job_settings["phenotype_col"] in dataio.columns(table) and not args.get("overwrite"):
            raise ValueError("That output column already exists. Choose a new name or tick Overwrite existing results.")
        job_settings["revision"] = tasks.source_revision(job_settings)
        job = jobs.start(root, "leiden", job_settings, request_id=request_id)
        return (
            spec,
            [],
            {"job": job},
            {"kind": "none"},
            "Leiden is running in the background; " + ("the selected output and its clusters will be replaced when it finishes." if args.get("overwrite") else "the clusters are saved in a new column."),
        )
    if name in ("run_som", "run_vortex"):
        from . import clustering

        method = name[len("run_") :]
        label = clustering.METHODS[method]["name"]
        values = clustering.values(root, method)
        _check_settings_revision(args, values)
        from .phenotyping_inputs import column_roles
        roles = column_roles(root, table)
        values = {**values, "cell_id_col": roles["cell_id"] or values["cell_id_col"]}
        if not values["markers"]:
            raise ValueError(f"Choose the markers for {label} and save the settings first.")
        bc = ((read_config(root).get("tab_configurations") or {}).get("tab6_batch_correction") or {}).get("bc_config")
        job_settings = {
            "table": table,
            "markers": list(values["markers"]),
            "output_col": values["output_col"] + "_clusters",
            "phenotype_col": values["output_col"],
            "cell_id_col": values["cell_id_col"] or "cell_id",
            "project_root": str(root),
            "config": values,
            "method": method,
            "overwrite_columns": [values["output_col"], values["output_col"] + "_clusters"] if args.get("overwrite") else [],
            "roi_col": roles["roi"],
            "cofactor": float((bc if isinstance(bc, dict) else {}).get("arcsinh_cofactor", 5.0) or 5.0),
        }
        tasks.validate_table(job_settings)
        if job_settings["phenotype_col"] in dataio.columns(table) and not args.get("overwrite"):
            raise ValueError("That output column already exists. Choose a new name or tick Overwrite existing results.")
        job_settings["revision"] = tasks.source_revision(job_settings)
        job = jobs.start(root, method, job_settings, request_id=request_id)
        return (
            spec,
            [],
            {"job": job},
            {"kind": "none"},
            f"{label} is running in the background; the groups are saved in a new column.",
        )
    if name == "run_neighbourhoods":
        from . import neighbourhoods

        chosen = neighbourhoods.values(read_config(root))
        _check_settings_revision(args, chosen)
        chosen = neighbourhoods.check(chosen)
        available = dataio.columns(table)
        if chosen["phenotype_col"] not in available:
            raise ValueError("Choose the phenotype column for the neighbourhoods and save the settings first.")
        chosen.update({key: chosen[key] if chosen[key] in available else "" for key in ("batch_col", "roi_col")})
        job_settings = {
            "table": table,
            "markers": [chosen["phenotype_col"], "x_centroid", "y_centroid"],
            "output_col": chosen["output_col"] + "_cn",
            "names_col": chosen["output_col"],
            "cell_id_col": "cell_id",
            "result": "neighbourhoods",
            "overwrite_columns": [chosen["output_col"], chosen["output_col"] + "_cn"] if args.get("overwrite") else [],
            "values": chosen,
            "images": _table_images(table, args.get("images")),
            "image_col": IMAGE_COLUMN,
        }
        tasks.validate_table(job_settings)
        job_settings["revision"] = tasks.source_revision(job_settings)
        job = jobs.start(root, "neighbourhoods", job_settings, request_id=request_id)
        return (
            spec,
            [],
            {"job": job},
            {"kind": "none"},
            "Finding the neighbourhoods" + (f" of {len(job_settings['images'])} image{'s' if len(job_settings['images']) != 1 else ''}"
                                             if job_settings["images"] else "") + " in the background; " +
            ("existing groups and names are replaced when it succeeds." if args.get("overwrite") else "they are saved in a new column."),
        )
    done = jobs.status(root, args["job_id"])
    result = done.get("result") or {}
    if (
        done.get("kind") not in {"leiden", "neighbourhoods", "som", "vortex", "subphenotype"}
        or done.get("status") != "done"
    ):
        raise ValueError("Choose a finished clustering or neighbourhood job.")
    if result.get("table") != table:
        raise ValueError("Choose this job's combined table at the top of Phenotyping to name its clusters.")
    # An old run must not rename a column whose cluster membership has been replaced.
    for newer in jobs.recent(root, limit=300):
        candidate = newer.get("result") or {}
        if newer.get("status") == "done" and candidate.get("clusters") and candidate.get("table") == table and candidate.get("column") == result["column"]:
            if newer["id"] != done["id"]:
                raise ValueError("These clusters have been replaced. Name the latest clustering run instead.")
            break
    # Naming edits update this run’s chosen output; unrelated existing columns stay protected.
    from .cluster_names import saved_names, owns_output
    known = {item["name"] for item in result.get("clusters") or []}
    clear = set(args.get("clear") or [])
    if clear - known:
        raise ValueError("Unknown clusters: " + ", ".join(sorted(clear - known)))
    changed_names = leiden.check_names(args["names"], result.get("clusters") or []) if args.get("names") else {}
    if clear & changed_names.keys():
        raise ValueError("Choose either a new name or clear for each cluster.")
    if not changed_names and not clear and not args.get("replace"):
        raise ValueError("Give cluster names to apply, or choose names to clear.")
    names = {} if args.get("replace") else dict(saved_names(root, table, result["column"], cluster_job_id=done["id"])["names"])
    names.update(changed_names)
    for cluster in clear:
        names.pop(cluster, None)
    output, taken = result["phenotype_column"], set(dataio.columns(table))
    # A second edit must not start from a half-written version or reuse its column.
    if any(job.get("kind") == "name_clusters" and job.get("status") in {"starting", "running", "stopping"}
           for job in jobs.recent(root, limit=100)):
        raise ValueError("The previous cluster naming job is still running. Wait for it to finish, then apply this edit.")
    replacing = owns_output(root, table, done["id"], output)
    if output in taken and not replacing:
        raise ValueError(f"{output} belongs to another result. Choose another Save as name and rerun, or rerun with Overwrite existing results.")
    job_settings = {
        "table": table,
        "markers": result.get("markers") or [],
        "output_col": output,
        "clusters_col": result["column"],
        "cell_id_col": result.get("cell_id_column") or "cell_id",
        "names": names,
        "cluster_job_id": args["job_id"],
        "overwrite_columns": [output] if replacing else [],
    }
    if done["kind"] == "neighbourhoods":
        job_settings["result"] = "neighbourhoods"
    tasks.validate_table(job_settings)
    job_settings["revision"] = tasks.source_revision(job_settings)
    job = jobs.start(root, "name_clusters", job_settings, request_id=request_id)
    kept = f" ({result['phenotype_column']} is kept)" if output != result["phenotype_column"] else ""
    return spec, [], {"job": job}, {"kind": "none"}, f"Saving the cluster names as {output}{kept}."


def _consensus(channel, args, spec, table, request_id):
    """Tab 7's Consensus ML through Studio, as a job; its four result columns wait beside the table."""
    from . import consensus, jobs, tasks

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project before running a job.")
    root = project["root"]
    chosen = consensus.values(root)
    _check_settings_revision(args, chosen)
    from .phenotyping_inputs import column_roles
    roles = column_roles(root, table)
    available = dataio.columns(table)
    consensus.check_run(chosen, available)
    output = chosen["output_phenotype_col"]
    job_settings = {
        "table": table,
        "markers": list(chosen["markers"]),
        "output_col": output,
        "cell_id_col": roles["cell_id"] or "cell_id",
        "project_root": str(root),
        "config": chosen,
        "auxiliary_columns": [output + suffix for suffix in ("_score", "_consensus", "_refined")],
        "overwrite_columns": [output + suffix for suffix in ("", "_score", "_consensus", "_refined")] if args.get("overwrite") else [],
    }
    tasks.validate_table(job_settings)
    job_settings["revision"] = tasks.source_revision(job_settings)
    job = jobs.start(root, "consensus_ml", job_settings, request_id=request_id)
    kept = f" ({chosen['output_phenotype_col']} is taken)" if output != chosen["output_phenotype_col"] else ""
    return (
        spec,
        [],
        {"job": job},
        {"kind": "none"},
        f"Consensus ML is running in the background; the results go to {output}{kept}.",
    )


def _background_project(channel):
    from spacemap_gui.studio.settings import read_config

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    return root, config, {**project.get("mappings", {}), **config.get("directory_mappings", {})}


def _open_background_image(channel, args, undo):
    """An image of the Background step, with its annotation file as a layer (to measure inside a class)."""
    from spacemap_gui.viewer.api import _annotation_layer
    from spacemap_gui.viewer.ui import window_image_spec, window_images

    from . import background

    root, config, mappings = _background_project(channel)
    plan = background.images_plan(root, mappings, config)
    row = next((row for row in plan["rows"] if row["image"] == args["image"]), None)
    if row is None:
        raise ValueError(f"{args['image']} is not in the images folder.")
    path = (Path(plan["folders"]["images"]) / row["image"]).resolve()
    window_images(channel)
    entries = registry.session_images(channel) or []
    image_id = next((i for i, entry in enumerate(entries) if Path(entry["path"]).resolve() == path), None)
    if image_id is None:
        raise ValueError(f"{row['image']} could not be listed for the viewer.")
    spec = window_image_spec(channel, image_id)
    data = {"image": row["image"], "annotation": None, "class": row["class"]}
    if row["annotation"]:
        image = next(layer for layer in spec["layers"] if layer["type"] == "image")
        layer = _annotation_layer(str(Path(plan["folders"]["annotations"]) / row["annotation"]), str(path), image["token"], row["annotation"])
        layer["open"] = True  # shown with all its classes, so the areas can be seen live
        spec["layers"] = [item for item in spec["layers"] if item.get("id") != layer["id"]] + [layer]
        data["annotation"] = layer["id"]
    return spec, [], data, undo, f"{row['image']}" + (f" with {row['annotation']}" if row["annotation"] else "") + "."


def _detect_background(channel, args, spec, request_id):
    """Auto background levels for the listed images, saved as their threshold tables, as a job."""
    from spacemap.background_detection import THRESHOLD_METHODS

    from . import background, jobs

    if args["method"] not in THRESHOLD_METHODS:
        raise ValueError("Choose one of Tab 2's methods: " + ", ".join(THRESHOLD_METHODS) + ".")
    root, config, mappings = _background_project(channel)
    plan = background.images_plan(root, mappings, config)
    rows = {row["image"]: row for row in plan["rows"]}
    unknown = [name for name in args["images"] if name not in rows]
    if unknown:
        raise ValueError("Not in the images folder: " + ", ".join(unknown[:4]))
    places = {kind: Path(folder) for kind, folder in plan["folders"].items()}
    items = [{
        "image": str(places["images"] / name), "root": str(root), "method": args["method"],
        "parameters": {"percentile": args.get("percentile",95), "manual": args.get("manual",0)}, "native": args.get("native",False),
        "annotation": str(places["annotations"] / rows[name]["annotation"]) if args["inside"] and rows[name]["annotation"] and rows[name]["class"] else None,
        "class": rows[name]["class"],
    } for name in dict.fromkeys(args["images"])]
    job = jobs.start(str(root), "detect_background", {"images": items}, request_id=request_id)
    inside = sum(1 for item in items if item["annotation"])
    return (spec, [], {"job": job}, {"kind": "none"},
            f"Detecting the background of {len(items)} image{'s' if len(items) > 1 else ''} with {args['method']}"
            + (f" ({inside} inside their annotation class)" if inside else " on the whole image")
            + ". The levels are kept in the project; Export writes them out.")


def _export_background(channel, args, spec, request_id):
    """Thresholds (CSV) and/or background-removed images (OME-TIFF) for the listed images, as a job."""
    from . import background, jobs

    if not args["csv"] and not args["write_image"]:
        raise ValueError("Tick Thresholds (CSV) and/or Image (OME-TIFF).")
    root, config, mappings = _background_project(channel)
    plan = background.images_plan(root, mappings, config)
    rows = {row["image"]: row for row in plan["rows"]}
    unknown = [name for name in args["images"] if name not in rows]
    if unknown:
        raise ValueError("Not in the images folder: " + ", ".join(unknown[:4]))
    places = {kind: Path(folder) for kind, folder in plan["folders"].items()}
    current = args.get("levels") or {}  # the open image's levels as shown (its latest edits)
    items, empty = [], []
    for name in dict.fromkeys(args["images"]):
        markers = background.image_markers(root, mappings, config, name)
        levels = current if name == args.get("open") and current else background.stored_levels(root, places['images'] / name) or (markers["levels"] or {})
        blanks = {m["name"]: m["subtract"] for m in markers["markers"] if m["subtract"]} if args["use_blanks"] else {}
        if (args["csv"] and not levels) or (args["write_image"] and not levels and not blanks):
            empty.append(name)
            continue
        items.append({
            "image": str(places["images"] / name), "output": str(background.output_path(places["output"], name)),
            "table": str(background.table_path(places["tables"], name)), "csv": args["csv"], "write_image": args["write_image"],
            "levels": levels if args["use_levels"] or args["csv"] else {}, "blanks": blanks, "method": args.get("method") or "Studio",
            "keep": [m["name"] for m in markers["markers"] if m["keep"]], "rename": {m["name"]: m["rename"] for m in markers["markers"] if m["rename"]},
            "colors": {m["name"]: m["new_color"] for m in markers["markers"] if m["new_color"]},
        })
    if not items:
        raise ValueError("Nothing to export: detect the levels first" + (" (or choose blanks)." if args["write_image"] else "."))
    job = jobs.start(str(root), "export_background", {"images": items}, request_id=request_id)
    what = " and ".join(x for x, on in (("thresholds (CSV)", args["csv"]), ("images (OME-TIFF)", args["write_image"])) if on)
    return (spec, [], {"job": job}, {"kind": "none"},
            f"Exporting {what} for {len(items)} image{'s' if len(items) > 1 else ''} in the background"
            + (f"; skipped (no levels yet): {', '.join(empty[:4])}" if empty else "") + ".")


def _compare_pair(channel, args, undo):
    """A pair's CODEX image with its raw H&E over it (layer "he", placed by the pair's pixel sizes and rotation)."""
    from spacemap.viewer.sources import image_info
    from spacemap_gui.studio.settings import read_config
    from spacemap_gui.viewer.ui import image_layer, window_image_spec, window_images

    from . import registration

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    details = registration.pair_markers(root, mappings, config, args["sample"])
    codex_dir, he_dir, _ = registration.folders(root, mappings)
    row = next(row for row in registration.plan(root, mappings, config)["rows"] if row["sample"] == args["sample"])
    if not row["he"]:
        raise ValueError(f"{args['sample']} has no H&E image: pick one in the pairs list.")
    codex, he = (codex_dir / row["codex"]).resolve(), (he_dir / row["he"]).resolve()
    window_images(channel)
    entries = registry.session_images(channel) or []
    image_id = next((i for i, entry in enumerate(entries) if Path(entry["path"]).resolve() == codex), None)
    if image_id is None:
        raise ValueError(f"{row['codex']} is not among the project's images.")
    spec = window_image_spec(channel, image_id)
    layer = image_layer(str(he), name="H&E to align · " + row["he"], opacity=0.6, visible=True, layer_id="he", save_display=False)
    if not image_info(str(he)).rgb:
        layer["channels"] = [{"index": i, "color": colour, "visible": True, "low": 0, "high": 255}
                             for i, colour in enumerate(("#ff0000", "#00ff00", "#0000ff"))]
    saved = (registration._tab1(config).get(registration.PAIR_MARKERS) or {}).get(row["codex"]) or {}
    layer["align"] = {"sample": args["sample"], "codex_mpp": details["mpp"]["codex"]["used"], "he_mpp": details["mpp"]["he"]["used"],
                      "rotation": float(saved.get("he_rotation", 0) or 0), "x": 0.0, "y": 0.0, "scaled": True}
    spec["layers"] = [item for item in spec["layers"] if item.get("id") != "he"] + [layer]
    return spec, [], None, undo, f"{args['sample']}: the H&E over the CODEX. Turn it, scale it by the pixel sizes and move it in Compare."


def _show_region(channel, spec, image, undo):
    """Add the open ROI's annotation file (matched by name, or chosen by hand) to the view."""
    from spacemap_gui.studio.settings import read_config
    from spacemap_gui.viewer.api import _annotation_layer

    from . import segmentation

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    path = registry.resolve(image["token"])
    settings = segmentation.values(root, read_config(root))
    found = segmentation.regions(settings, [Path(path).name])["rows"][0]
    if not found["annotation"]:
        raise ValueError(f"No annotation file for {Path(path).name}: pick one in Where to segment.")
    file = str(Path(settings["tab4_annot_geojson_dir"]) / found["annotation"])
    layer = _annotation_layer(file, path, image["token"], found["annotation"])
    spec["layers"] = [item for item in spec["layers"] if item.get("id") != layer["id"]] + [layer]
    chosen = settings["tab4_mask_classes"]
    return spec, [], None, undo, f"Showing {found['annotation']}" + (f"; Cellpose segments inside {', '.join(chosen)}." if chosen else ".")


def _tag_cells(channel, args, spec, request_id):
    """Tab 5's cell tagging through Studio, as a job: every listed table (or all) with its own annotation."""
    from spacemap_gui.studio.settings import read_config
    from spacemap_gui.tabs import cell_tagging_ui as tab5

    from . import jobs, tagging

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project before running a job.")
    root = project["root"]
    state = tagging.values(root, read_config(root))
    found = tagging.sources(root, state)
    if not found["rows"]:
        raise ValueError("No cell tables found: choose the folder that holds them.")
    if not found["coordinates"]["x"] or not found["coordinates"]["y"]:
        raise ValueError("Choose the columns that hold the cell coordinates.")
    if not any(row["annotation"] for row in found["rows"]):
        raise ValueError("No cell table has annotations: export them in the Annotation step, or pick a file for each table.")
    engine_rules, edge, *_ = tab5.build_engine_inputs(state)
    beds = tab5.selected_bed_labels(state=state)
    if not any(rule.is_active() for rule in engine_rules) and edge is None and not beds:
        raise ValueError("Choose at least one column to add for an annotation, then save the tagging settings.")
    tables = list(args.get("tables") or [])
    if tables and found["mode"] == tab5.MODE_PER_IMAGE:
        unknown = [name for name in tables if name not in found["tables"]]
        if unknown:
            raise ValueError("Not a cell table here: " + ", ".join(unknown[:3]))
    count = len(tables) if tables and found["mode"] == tab5.MODE_PER_IMAGE else len(found["rows"])
    exports, _ = tagging._exports_and_images(root)
    job = jobs.start(root, "cell_tagging", {"state": state, "found": found, "tables": tables, "root": str(root), "exports": exports},
                     request_id=request_id)
    where = "the combined table" if found["mode"] == tab5.MODE_MERGED else f"{count} cell table{'s' if count != 1 else ''}"
    return (spec, [], {"job": job}, {"kind": "none"}, f"Tagging the cells of {where} in the background; the originals are kept.")


IMAGE_COLUMN = "batch_id"


def _table_images(table, images):
    """The images (values of the table's image column) a spatial job is limited to; none: every image."""
    if not images:
        return []
    if IMAGE_COLUMN not in dataio.columns(table):
        raise ValueError(f"The combined table has no {IMAGE_COLUMN} column to choose images by.")
    return [str(name) for name in dict.fromkeys(images)]


def _interactions(channel, args, spec, table, request_id):
    """Tab 8's cell-cell interaction test through Studio, as a job; the result stays with the job."""
    from spacemap_gui.studio.settings import read_config

    from . import interactions, jobs

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project before running a job.")
    root = project["root"]
    chosen = interactions.values(read_config(root))
    available = dataio.columns(table)
    missing = [c for c in ("x_centroid", "y_centroid") if c not in available]
    if chosen["phenotype_col"] not in available:
        raise ValueError("Choose the phenotype column for cell-cell interactions and save the settings first.")
    if missing:
        raise ValueError("The combined table has no cell positions (" + ", ".join(missing) + ").")
    images = _table_images(table, args.get("images"))
    job = jobs.start(root, "cell_interactions", {"table": table, "values": chosen, "images": images, "image_col": IMAGE_COLUMN}, request_id=request_id)
    return (
        spec,
        [],
        {"job": job},
        {"kind": "none"},
        f"Testing which {chosen['phenotype_col']} phenotypes are neighbours within {chosen['radius_um']:g} µm"
        + (f" in {len(images)} image{'s' if len(images) != 1 else ''}" if images else "") + ", in the background.",
    )


def _subphenotyping(channel, args, spec, table, request_id):
    """Tab 7's sub-phenotyping through Studio: find the markers inside each phenotype, or split them, as a job."""
    from . import jobs, subphenotypes, tasks

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project before running a job.")
    root = project["root"]
    chosen = subphenotypes.values(root)
    _check_settings_revision(args, chosen)
    from .phenotyping_inputs import column_roles
    roles = column_roles(root, table)
    available = dataio.columns(table)
    if chosen["parent_col"] not in available:
        raise ValueError("Choose the phenotype column to split and save the settings first.")
    if not chosen["parent_values"]:
        raise ValueError("Choose the phenotypes to split.")
    missing = [m for m in chosen["markers"] if m not in available]
    if not chosen["markers"] or missing:
        raise ValueError("Choose markers the table has" + (f" (no {', '.join(missing[:5])})." if missing else "."))
    mode = args.get("mode") or "run"
    names = chosen["studio_output_col"]
    job_settings = {
        "table": table,
        "markers": list(chosen["markers"]),
        "output_col": names + "_clusters",
        "phenotype_col": names,
        "cell_id_col": roles["cell_id"] or "cell_id",
        "project_root": str(root),
        "config": chosen,
        "mode": mode,
        "overwrite_columns": [names, names + "_clusters"] if args.get("overwrite") and mode == "run" else [],
    }
    if mode == "run":
        if names in available and not args.get("overwrite"):
            raise ValueError("That output column already exists. Choose a new name or tick Overwrite existing results.")
        tasks.validate_table(job_settings)
    job_settings["revision"] = tasks.source_revision(job_settings)
    job = jobs.start(root, "subphenotype", job_settings, request_id=request_id)
    message = (
        "Finding the markers that vary inside each phenotype, in the background."
        if mode == "discover"
        else f"Splitting {', '.join(chosen['parent_values'][:4])} in the background; the sub-clusters go to {names}_clusters."
    )
    return spec, [], {"job": job}, {"kind": "none"}, message


def _change_settings(channel, args, spec):
    """A step's saved settings changed through the step's own checks (the ones its Run and the classic tab use)."""
    from . import step_settings

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    method = args["method"]
    previous, now = step_settings.change(project["root"], method, args["changes"])
    undo = {"kind": "settings", "root": str(project["root"]), "method": method, "previous": previous, "expected": now}
    shown = "; ".join(f"{key} {json.dumps(value, default=str)[:80]}" for key, value in now.items())
    return (
        spec,
        [],
        {"settings": {"method": method, "changes": now}},
        undo,
        f"Saved the {method.replace('_', ' ')} settings: {shown}.",
    )


def _correct(channel, spec, request_id):
    """Tab 6's batch correction with its saved settings and markers, as a job."""
    from spacemap_gui.studio.settings import read_config

    from . import correction, jobs

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    plan = correction.plan(root, mappings, config)
    if not plan["tables"] or not plan["markers"]:
        raise ValueError(
            "Batch correction needs the ROIs' cell tables and the markers to correct; open its step first."
        )
    # The settings the Batch correction step shows (its mode when none was chosen yet included).
    settings = correction.job_settings(
        root, mappings, {**correction.config_of(config), **plan["settings"]}, plan["markers"]
    )
    job = jobs.start(str(root), "correct_batches", settings, request_id=request_id)
    return (
        spec,
        [],
        {"job": job},
        {"kind": "none"},
        f"Correcting {len(plan['markers'])} markers across {plan['tables']} ROI tables in the background; "
        "the corrected table is a new file.",
    )


def _train_cellpose(channel, args, spec, request_id):
    """The next version of the session's Cellpose model, trained on the corrected training tiles, as a job."""
    from spacemap_gui.studio.settings import read_config

    from . import jobs, segmentation, training

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = project["root"]
    found = training.examples(root)
    if not found["session"]:
        raise ValueError("Start a training session first: name the model and choose where it starts from.")
    if not found["counts"]["train"]:
        raise ValueError("Keep at least one corrected training tile first.")
    settings = {
        "project_root": str(root),
        "n_epochs": args["epochs"],
        "gpu": bool(segmentation.values(root, read_config(root))["tab4_gpu"]),
    }
    job = jobs.start(root, "train_cellpose", settings, request_id=request_id)
    version = f"{found['session']['name']}_v{len(found['session']['versions']) + 1}"
    return spec, [], {"job": job}, {"kind": "none"}, f"Training {version} in the background."


def _classes(channel, spec, image, request_id):
    """Tab 4 step 3 through Studio: export the open image's named classes, as a job."""
    from spacemap_gui.studio.settings import read_config

    from . import annotation, jobs

    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    _, output = annotation.folders(root, mappings)
    path = Path(registry.resolve(image["token"]))
    sample = annotation.sample_id(path.name)
    tab4 = (config.get("tab_configurations") or {}).get("tab4_annotation") or {}
    if (tab4.get("annot_ta_domain_map_image") or {}).get("sample") != sample:
        raise ValueError("Save the class names for this image first.")
    settings = {
        "image": str(path),
        "sample": sample,
        "output": str(output),
        "config": {"tab_configurations": {"tab4_annotation": tab4}},
    }
    from . import annotation_inputs
    settings["scope"] = str(annotation_inputs.scope_path(root, path, output, "classes"))
    job = jobs.start(str(root), "export_classes", settings, request_id=request_id)
    return (
        spec,
        [],
        {"job": job},
        {"kind": "none"},
        f"Exporting the classes and domains of {sample} in the background; Tab 4's class files are written.",
    )


def _undo(channel, command_id, spec):
    with _database() as db:
        row = db.execute(
            "SELECT undo,undone,result FROM commands WHERE channel=? AND id=?", (channel, command_id)
        ).fetchone()
    if not row or row[1]:
        raise ValueError("That action is not available to undo.")
    undo = json.loads(row[0])
    original = json.loads(row[2])
    effects = []
    restored = spec
    if undo["kind"] == "none":
        raise ValueError("This action did not change anything that can be undone.")
    if undo["kind"] == "view":
        restored = undo["spec"]
        effects = [{"type": "restore_view", "spec": deepcopy(restored)}]
    elif undo["kind"] == "selection_labels":
        if fingerprint(undo["table"]) != undo["fingerprint"]:
            raise ValueError("The cell table changed. Review the current cells before editing their labels.")
        reference.undo_groups(
            undo["table"], undo["column"], undo["previous"], undo["expected"], "undo-" + command_id, **undo["keys"]
        )
        effects = [{"type": "refresh_labels"}]
    elif undo["kind"] == "labels":
        if fingerprint(undo["table"]) != undo["fingerprint"]:
            raise ValueError("The cell table changed. Review the current cells before editing their labels.")
        reference.save(
            undo["table"],
            undo["roi"],
            undo["column"],
            {int(k): v for k, v in undo["previous"].items()},
            "undo-" + command_id,
            source="undo",
            expected={str(k): v for k, v in undo["expected"].items()},
            **undo["keys"],
        )
        effects = [{"type": "refresh_labels"}]
    elif undo["kind"] == "threshold":
        session = live_resolve.open_session(undo["resolve"])
        if session.thresholds[undo["marker"]] != undo["expected"]:
            raise ValueError("The threshold changed again. Keep the newer value or adjust it directly.")
        live_resolve.assign(session, {undo["marker"]: undo["previous"]})
        effects = [{"type": "refresh_resolve"}]
    elif undo["kind"] == "settings":
        from . import step_settings

        now = step_settings.current(undo["root"], undo["method"])
        if any(now.get(key) != value for key, value in undo["expected"].items()):
            raise ValueError("These settings changed again. Their newer values were kept.")
        step_settings.change(undo["root"], undo["method"], undo["previous"])
    elif undo["kind"] == "file":
        path = Path(undo["path"])
        if not path.exists() or fingerprint(path) != undo["expected"]:
            raise ValueError("The saved file changed again. Its newer contents were kept.")
        if undo["previous"] is None:
            path.unlink()
        else:
            temporary = path.with_name(path.name + ".undo-" + uuid.uuid4().hex)
            temporary.write_text(undo["previous"])
            os.replace(temporary, path)
    with _database() as db:
        db.execute("UPDATE commands SET undone=1 WHERE channel=? AND id=?", (channel, command_id))
    return restored, effects, None, {"kind": "none"}, "Undid " + original["name"].replace("_", " ") + "."


def execute(channel, body):
    """Checked command entry point. No arbitrary paths, functions or code execute."""
    if not isinstance(body, dict):
        raise ValueError("Provide a named Studio action.")
    view = registry.session_view(channel, mark_seen=False)
    if view is None:
        raise ValueError("Open Studio from SPACEMAP to link this session.")
    project = registry.session_project(channel) or {}
    name = body.get("name")
    args = body.get("args", {})
    request_id = body.get("request_id") or uuid.uuid4().hex
    if (
        not isinstance(request_id, str)
        or not 8 <= len(request_id) <= 80
        or not all(c.isalnum() or c in "-_" for c in request_id)
    ):
        raise ValueError("Use a valid request number.")
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    cpu = time.process_time()
    signature = _signature(name, args)
    with _lock(channel):
        receipt = _receipt(channel, request_id, signature)
        if receipt:
            latest = registry.session_view(channel, mark_seen=False)
            if latest["version"] != receipt.get("version"):
                receipt = {**receipt, "spec": latest["spec"], "version": latest["version"], "effects": []}
            return {**receipt, "replayed": True}
        result = {"name": name, "command_id": request_id, "corrections": []}
        try:
            args = validate(name, args)
            if name == "assign_selection" and not args.get("column"):
                args["column"] = view["spec"].get("reference_column") or "phenotype_ref1"
            changes_project = COMMANDS[name].changes_project
            if name == "undo":
                with _database() as db:
                    stored = db.execute(
                        "SELECT undo FROM commands WHERE channel=? AND id=?", (channel, args["command_id"])
                    ).fetchone()
                changes_project = bool(
                    stored and json.loads(stored[0])["kind"] in {"labels", "selection_labels", "file", "settings"}
                )
            if changes_project and not body.get("confirmed") is True:
                result.update(status="needs_confirmation", message=COMMANDS[name].description, args=args)
                if name == "assign_selection":
                    result["message"] = (
                        f"Assign {view['spec'].get('selection', {}).get('count', 0):,} selected cells to {args['phenotype']} in {args['column']}."
                    )
            elif changes_project and project.get("read_only"):
                raise ValueError("This project is open for viewing only.")
            else:
                latest = registry.session_view(channel, mark_seen=False)
                if body.get("version") is not None and body["version"] != latest["version"]:
                    raise ValueError("The view changed. Try again using the current image.")
                before = _snapshot(deepcopy(latest["spec"]), body.get("view"))
                spec, effects, data, undo, message = _execute(
                    channel, name, args, deepcopy(before), result["corrections"], request_id
                )
                # Polling/reloading clients must distinguish opening an image
                # from an explicit request to change its display.
                event = {"id": request_id, "name": name, "effects": effects}
                changed = spec != latest["spec"] or bool(effects)
                if changed:
                    spec["command"] = event
                    json.dumps(spec, allow_nan=False)
                    version = registry.publish(channel, spec, project=project)
                else:
                    version = latest["version"]
                result.update(
                    status="done",
                    message=message,
                    effects=effects,
                    data=data,
                    spec=spec,
                    version=version,
                    undo=undo["kind"] != "none",
                    changes_project=changes_project,
                )
                _record(channel, request_id, signature, result, undo)
        except NameChoice as exc:
            result.update(
                status="needs_choice", message=str(exc), argument=exc.kind, requested=exc.value, choices=exc.choices
            )
        except (ValueError, KeyError, TypeError) as exc:
            result.update(status="error", message=str(exc))
        _journal(
            project, channel, str(name), args, result, started, time.perf_counter() - clock, time.process_time() - cpu
        )
        return result
