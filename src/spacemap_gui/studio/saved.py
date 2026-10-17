"""Jobs > Saved results: what Studio has saved, and writing it into the table only when the user asks.

Studio never rewrites a CSV on its own. Its results are kept beside the combined table
(``spacemap.pending_columns``: one small Parquet file per column; labels in the labels store) and
read with the table at once. Tables Studio makes (batch correction) are Parquet files. From here the
user can:

* **Write into the table**: the pending columns and labels go into the combined table (the CSV, or
  the Parquet table) in one pass, in the background (classic's own write, ``start_commit``).
* **Discard** pending columns, or **re-apply** them by cell ID after the table was rewritten.
* **Save a CSV copy** of a Parquet table (with its pending columns), for classic SPACEMAP.
"""

from pathlib import Path


def _tables(project):
    """Only project-created tables and the explicitly loaded Parquet table."""
    from .correction import output_dir

    paths = []
    table = project.get("unified")
    if table and Path(table).suffix.lower() == ".parquet" and Path(table).is_file():
        paths.append(Path(table).resolve())
    if project.get("root"):
        folder = output_dir(project["root"], project.get("mappings") or {})
        paths.extend(path.resolve() for path in sorted(folder.glob("*.parquet"),
                     key=lambda item: item.stat().st_mtime, reverse=True)
                     if path.is_file() and not path.name.endswith(".csv.parquet"))
    return list(dict.fromkeys(paths))


def state(project):
    from spacemap import labels, pending_columns

    table = project.get("unified")
    found = {"table": "", "pending": [], "labels": 0, "writing": {"state": "idle"}, "tables": []}
    if table and Path(table).is_file():
        found.update(
            table=Path(table).name,
            parquet=Path(table).suffix.lower() == ".parquet",
            pending=[
                {key: item.get(key) for key in ("column", "rows", "saved_at", "source", "stale")}
                for item in pending_columns.pending(table)
            ],
            labels=labels.pending_count(table),
            writing=pending_columns.commit_status(table),
        )
    for path in _tables(project):
        csv = path.with_suffix(".csv")
        found["tables"].append(
            {"name": path.name, "id": str(path),
             "megabytes": round(path.stat().st_size / 1e6, 1),
             "in_use": bool(table) and Path(table).resolve() == path,
             "csv": csv.name if csv.is_file() else ""})
    return found


def act(project, action, body):
    from spacemap import pending_columns

    from . import jobs
    table = project.get("unified")
    if action in {"write", "discard", "reapply"}:
        if not table or not Path(table).is_file():
            raise ValueError("No combined table yet.")
        if pending_columns.commit_status(table).get("state") == "writing":
            raise ValueError("The table is being written; wait for it to finish.")
    if action == "write":
        pending_columns.start_commit(table)
    elif action == "discard":
        columns = body.get("columns")
        if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
            raise ValueError("Choose the saved columns to discard.")
        known = {item["column"] for item in pending_columns.pending(table)}
        if set(columns) - known:
            raise ValueError("Those columns are not waiting to be written.")
        pending_columns.discard(table, columns)
    elif action == "reapply":
        pending_columns.reapply_by_id(table)
    elif action == "export":
        name = str(body.get("name") or "")
        choices = [path for path in _tables(project) if path.name == name
                   and (not body.get("table") or str(path) == body["table"])]
        if len(choices) != 1:
            raise ValueError("Choose one of the listed tables.")
        path = choices[0]
        return {"job": jobs.start(project["root"], "export_table", {"table": str(path)})}
    else:
        raise ValueError("Unknown action.")
    return state(project)


def export(settings, folder, report):
    """The job: a CSV copy of a Parquet table (with its pending columns and labels), beside it."""
    from spacemap import dataio

    source = Path(settings["table"])
    report(f"Reading {source.name}…", 0.05)
    frame = dataio.load(source)
    target = source.with_suffix(".csv")
    report(f"Writing {target.name} ({len(frame):,} cells; the slow part)…", 0.2)
    temporary = target.with_name("." + target.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)
    return {"output": str(target), "name": target.name, "cells": int(len(frame))}
