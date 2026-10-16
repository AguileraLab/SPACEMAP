"""Cell data files (CSV, TSV, Parquet) shown on an image, as cell layers.

A table with one row per cell and its position becomes a cell layer, so it has every Cells control (outlines or
filled, Colour by, ranges, groups):

- **Colour this image's cells**: linked to the image's segmentation by cell number (the table's label column) or, when
  the numbers belong to another run, by position (each row goes to the cell its x, y falls in);
- **Circles**: one disc per row, of a chosen size;
- **Voronoi**: each row's nearest area, no farther than a chosen reach (for tables without a segmentation).

Rows can be kept to one ROI (a column naming the image) and to inside an annotation. What is built is kept per image
in ``outputs/studio/cell_data/<image-key>/``: the rows used (Parquet, with batch_id and label for the viewer), the
circles or Voronoi cells (GeoJSON, which can be exported), and an index so the image opens with them again.
"""

from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

TABLES = (".csv", ".tsv", ".txt", ".parquet")
SHOW = ("cells", "circles", "voronoi")
_X = ("x_centroid", "centroid_x", "x", "cell_x", "x_position", "pos_x", "centroid x px", "centroid x µm", "centroid x um",
      "x_um", "x (µm)", "x_px", "xcenter", "x_center")
_ROI = ("batch_id", "roi", "sample", "image", "image_id", "sample_id", "roi_id", "source", "file", "batch")
_LABEL = ("label", "cell_label", "cellid", "cell_id", "object_id", "id")


_X_PATTERN = re.compile(r"^(cell[ _]?|centroid[ _]?|pos(ition)?[ _]?)?x([ _]?(centroid|center|centre|pos(ition)?))?([ _(\[].*)?$", re.I)


def _y_of(name):
    """The y column's name for an x column's: the x that stands for the axis becomes y (X µm → Y µm)."""
    return re.sub(r"(?<![A-Za-z])([xX])(?![A-Za-z])", lambda m: "y" if m.group(1) == "x" else "Y", name, count=1)


def _guess(columns, wanted):
    low = {c.lower(): c for c in columns}
    for name in wanted:
        if name in low:
            return low[name]
    return ""


def _units(column):
    return "um" if re.search(r"µm|\bum\b|_um|micron", column, re.I) else "px"


def shown_name(path):
    """A table's name as the user gave it (a dropped file is kept under a prefixed name)."""
    path = Path(path)
    return re.sub(r"^[0-9a-f]{8}_", "", path.name) if "dropped" in path.parts else path.name


def columns_of(path):
    from spacemap import dataio

    return list(dataio.columns(str(path)))


def _read(path, columns=None, roi_column="", roi_value=None):
    """A table's rows (only one ROI's when asked), reading Parquet with a filter and CSV in chunks."""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        filters = [(roi_column, "=", roi_value)] if roi_column and roi_value is not None else None
        try:
            return pq.read_table(path, columns=columns, filters=filters).to_pandas()
        except Exception:  # noqa: BLE001 - a type mismatch in the filter: read, then filter as text
            frame = pq.read_table(path, columns=columns).to_pandas()
            return frame[frame[roi_column].astype(str) == str(roi_value)] if filters else frame
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    if not roi_column or roi_value is None:
        return pd.read_csv(path, sep=sep, usecols=columns, low_memory=False)
    parts = [chunk[chunk[roi_column].astype(str) == str(roi_value)]
             for chunk in pd.read_csv(path, sep=sep, usecols=columns, chunksize=500_000, low_memory=False)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def inspect(path, sample):
    """What the page needs to set a table up: its columns, the guessed position, ROI and label columns, and the ROI
    value that names this image."""
    from spacemap.viewer.catalog import sample_of

    columns = columns_of(path)
    x = _guess(columns, _X) or next((c for c in columns if _X_PATTERN.match(c.strip()) and _y_of(c) in columns), "")
    y = next((c for c in columns if c.lower() == _y_of(x).lower()), "") if x else ""
    roi = _guess(columns, _ROI)
    values, here = [], ""
    if roi:
        frame = _read(path, [roi])
        values = sorted(map(str, frame[roi].dropna().astype(str).unique()))[:2000]
        wanted = {sample, sample_of(sample)}
        here = next((v for v in values if v in wanted or sample_of(v) in wanted), "")
    return {"columns": columns, "x": x, "y": y, "units": _units(x) if x else "px", "roi_column": roi, "roi_values": values,
            "roi_value": here, "label_column": _guess(columns, _LABEL), "name": shown_name(path)}


def _folder(root, image):
    from .annotation_inputs import image_key

    return Path(root) / "outputs/studio/cell_data" / image_key(image)


def records(root, image):
    try:
        found = json.loads((_folder(root, image) / "index.json").read_text())
    except (OSError, ValueError):
        return []
    return [item for item in found if isinstance(item, dict) and Path(item.get("table", "")).is_file()]


def _save(root, image, items):
    from .annotation_edits import _atomic
    from .settings import project_lock

    with project_lock(root):
        _atomic(_folder(root, image) / "index.json", json.dumps(items, indent=2).encode())


def area_choices(root, config, image, output):
    """The annotations rows can be kept inside, grouped as the Annotation lists: [[group, [[key, name], ...]], ...]."""
    from . import annotation_names

    groups = {}
    for item in annotation_names.groups(root, config, image, output):
        if item["role"] == "good":
            continue
        groups.setdefault(item["source"] or "This image", []).append([item["key"], item["name"]])
    return [[group, items] for group, items in groups.items()]


def _area(root, config, image, output, key):
    """The union of the polygons of one annotation name (key from area_choices), in image pixels."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    from . import annotation, annotation_names

    group = next((g for g in annotation_names.groups(root, config, image, output) if g["key"] == key), None)
    if group is None:
        raise ValueError("That annotation is not on this image any more. Choose Within again.")
    entries = {e["path"]: e for e in annotation_names.files(root, config, image, output)}
    shapes = []
    for path in group["files"]:
        entry = entries.get(path)
        if entry is None:
            continue
        features, _ = annotation_names._read(entry, annotation.tissue_label(config))
        shapes += [shape(f["geometry"]) for f in features if annotation_names.name_of(f, entry["kind"], entry["layer"]) == group["name"]]
    if not shapes:
        raise ValueError(f"{group['name']} has no polygons.")
    return unary_union(shapes), group["name"]


def _labels_at(pyramid, xs, ys, block=4096):
    """The cell number under each point, read from the label pyramid's full-resolution level block by block."""
    from spacemap.viewer.sources import open_source

    source = open_source(str(pyramid))
    level = source.info.levels[0]
    out = np.zeros(len(xs), np.int64)
    cols, rows = (xs // block).astype(int), (ys // block).astype(int)
    inside = (xs >= 0) & (ys >= 0) & (xs < level.width) & (ys < level.height)
    for bx, by in set(zip(cols[inside].tolist(), rows[inside].tolist())):
        x0, y0 = bx * block, by * block
        x1, y1 = min(x0 + block, level.width), min(y0 + block, level.height)
        region = source.read_region(0, 0, x0, y0, x1, y1)
        pick = inside & (cols == bx) & (rows == by)
        out[pick] = region[(ys[pick] - y0).astype(int), (xs[pick] - x0).astype(int)]
    return out


def build(root, config, mappings, image, source, choice, cache, segmentation=None):
    """Build one data layer of *image* from the table *source* with the user's *choice*, and remember it.

    choice: x, y, units ("px" or "um"), roi_column, roi_value, label_column, show ("cells", "circles", "voronoi"),
    link ("label" or "position", for cells), size_um (circle diameter), reach_um (Voronoi), within (an annotation key
    or ""). Returns the record (with the table, the GeoJSON when drawn, and counts)."""
    import pandas as pd
    import shapely
    from shapely.geometry import box, mapping

    from spacemap.viewer import cells as viewer_cells
    from spacemap.viewer.sources import image_info

    from . import annotation

    show = choice.get("show")
    if show not in SHOW:
        raise ValueError("Show the data as this image's cells, circles or Voronoi cells.")
    columns = columns_of(source)
    for key in ("x", "y"):
        if choice.get(key) not in columns:
            raise ValueError("Choose the columns that hold the cells' x and y.")
    roi_column = choice.get("roi_column") or ""
    if roi_column and roi_column not in columns:
        raise ValueError("Choose the column that names the image, or none.")
    info = image_info(str(image))
    microns = info.pixel_size_um or 0.5
    frame = _read(source, None, roi_column, choice.get("roi_value") if roi_column else None).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows for this image: check the column that names it and its value.")
    scale = 1.0 / microns if choice.get("units") == "um" else 1.0
    xs = pd.to_numeric(frame[choice["x"]], errors="coerce").to_numpy(float) * scale
    ys = pd.to_numeric(frame[choice["y"]], errors="coerce").to_numpy(float) * scale
    keep = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (ys >= 0) & (xs < info.width) & (ys < info.height)
    rows_total = len(frame)
    within_name = ""
    area = None
    if choice.get("within"):
        output = annotation.folders(root, mappings)[1]
        area, within_name = _area(root, config, image, output, choice["within"])
        keep &= shapely.contains_xy(area, xs, ys)
    frame, xs, ys = frame[keep].reset_index(drop=True), xs[keep], ys[keep]
    if frame.empty:
        raise ValueError("No row falls inside the image" + (f" and {within_name}" if within_name else "") + ": check x, y and their units.")
    sample = Path(image).name
    from spacemap.viewer.catalog import sample_of

    roi = sample_of(sample)
    identity = sha256(json.dumps({"source": str(Path(source).resolve()), **choice}, sort_keys=True).encode()).hexdigest()[:12]
    folder = _folder(root, image)
    folder.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(shown_name(source)).stem)[:60]
    matched = len(frame)
    geojson = None
    if show == "cells":
        if not segmentation:
            raise ValueError("This image has no cell outlines to colour: show the data as circles or Voronoi cells.")
        if choice.get("link") == "label":
            label_column = choice.get("label_column")
            if label_column not in frame:
                raise ValueError("Choose the column with the cell numbers, or link by position.")
            labels = pd.to_numeric(frame[label_column], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        else:
            pyramid = viewer_cells.ensure_label_pyramid(segmentation, info.width, info.height, cache)
            labels = _labels_at(pyramid, xs, ys)
        matched = int((labels > 0).sum())
        frame = frame.assign(label=labels)[labels > 0]
    else:
        frame = frame.assign(label=np.arange(1, len(frame) + 1))
        points = shapely.points(xs, ys)
        bounds = box(0, 0, info.width, info.height)
        if show == "circles":
            radius = max(float(choice.get("size_um") or 10.0) / 2.0 / microns, 0.5)
            shapes = shapely.buffer(points, radius, quad_segs=4)
        else:
            reach = max(float(choice.get("reach_um") or 15.0) / microns, 1.0)
            regions = np.asarray(shapely.voronoi_polygons(shapely.multipoints(points), extend_to=bounds, ordered=True).geoms)
            if len(regions) != len(points):  # repeated positions: give each point the region it lies in
                tree = shapely.STRtree(regions)
                found = tree.query_nearest(points, return_distance=False)  # (point, region) pairs; ties give several
                region_of = np.zeros(len(points), int)
                region_of[found[0][::-1]] = found[1][::-1]  # the first pair of each point wins
                regions = regions[region_of]
            shapes = shapely.intersection(regions, shapely.buffer(points, reach, quad_segs=8))
        shapes = shapely.intersection(shapes, area if area is not None else bounds)
        features = [{"type": "Feature", "geometry": mapping(geometry), "properties": {"label": int(label)}}
                    for geometry, label in zip(shapes, frame["label"]) if not geometry.is_empty]
        geojson = folder / f"{stem}_{show}_{identity}.geojson"
        geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    if "batch_id" in frame:
        frame["batch_id"] = roi
    else:
        frame.insert(0, "batch_id", roi)
    table = folder / f"{stem}_{identity}.parquet"
    for column in frame.columns[frame.dtypes == object]:
        frame[column] = frame[column].astype("string")
    frame.to_parquet(table, index=False)
    record = {"id": identity, "name": shown_name(source), "source": str(Path(source).resolve()), "choice": choice, "show": show,
              "table": str(table), "geojson": str(geojson) if geojson else "", "rows": rows_total, "kept": int(keep.sum()),
              "matched": matched, "within": within_name, "within_key": choice.get("within") or ""}
    _save(root, image, [item for item in records(root, image) if item.get("id") != identity] + [record])
    return record


def remove(root, image, identity):
    found = records(root, image)
    gone = [item for item in found if item.get("id") == identity]
    if not gone:
        raise ValueError("That data layer is not on this image any more.")
    _save(root, image, [item for item in found if item.get("id") != identity])
    from spacemap.viewer import cells as viewer_cells

    building = {str(Path(thread.name)) for thread in viewer_cells._BUILDING.values() if thread.is_alive()}
    for item in gone:  # what was built for it (the source table is never touched)
        for path in (item.get("table"), item.get("geojson")):
            if not path or not Path(path).is_file():
                continue
            if path == item.get("geojson") and any(Path(path).stem in name for name in building):
                continue  # its outlines are still being drawn: the file goes with the next clean-up
            Path(path).unlink()


def describe(record):
    """In words: '12,345 of 13,000 rows · Voronoi · inside Tumor', for the layer's line."""
    show = {"cells": "this image's cells, by " + ("cell number" if record["choice"].get("link") == "label" else "position"),
            "circles": "circles", "voronoi": "Voronoi cells"}[record["show"]]
    count = record["matched"] if record["show"] == "cells" else record["kept"]
    return f"{count:,} of {record['rows']:,} rows · {show}" + (f" · inside {record['within']}" if record.get("within") else "")
