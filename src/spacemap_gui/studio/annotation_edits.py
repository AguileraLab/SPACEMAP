"""Editable annotation polygons and lossless, revision-checked local history.

Reads are read-only. Each mutation archives the full geometry before replacing
the GeoJSON atomically. Display copies carry the revision their geometry came
from, so Undo and renaming never replace a detailed polygon with its simplified
screen outline. The lock also covers background jobs and other viewer windows.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import uuid

from spacemap.viewer import shapes
from spacemap.sharing import open_lock

EXAMPLE_CLASSES = {
    "positive": {"name": "Artifact", "color": [230, 25, 75]},
    "negative": {"name": "Good tissue", "color": [46, 204, 113]},
}


def revision(path):
    path = Path(path)
    return sha256(path.read_bytes() if path.is_file() else b"").hexdigest()


def _home(path):
    path = Path(path)
    return path.parent / "studio_history" / path.name


@contextmanager
def locked(path):
    import fcntl

    home = _home(path)
    home.mkdir(parents=True, exist_ok=True)
    with open_lock(home / "edit.lock") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
        from spacemap.sharing import share
        share(path)
    finally:
        temporary.unlink(missing_ok=True)


def polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry] if not geometry.is_empty and geometry.area > 0 else []
    return [part for child in getattr(geometry, "geoms", []) for part in polygons(child)]


def prepare(features, *, kind="regions", label="tissue"):
    """Separate connected components without rounding, dropping holes or changing classic classes."""
    from shapely.geometry import mapping, shape
    from shapely.validation import make_valid

    found = []
    used = set()
    for index, feature in enumerate(features):
        source = feature["geometry"]
        if _CHECKED.get(id(source)) is source and kind != "shapes":  # a valid polygon already checked (read, then saved)
            geometry, valid = None, True
        else:
            geometry = shape(source)
            valid = geometry.is_valid
        if not valid:
            geometry = make_valid(geometry)
        props = dict(feature.get("properties") or {})
        if geometry is None and not props.get("id"):
            geometry = shape(source)
        # Stable for old files without IDs; opening the same file in two windows agrees.
        base_id = str(props.get("id") or sha256(geometry.wkb + str(index).encode()).hexdigest()[:20])
        if kind == "shapes":
            props["id"] = base_id
            props.pop("simplified", None)
            props.pop("_studio_geometry", None)
            found.append(shapes.normalise({**feature, "geometry": mapping(geometry), "properties": props}))
            continue
        classification = props.get("classification")
        if isinstance(classification, dict) and "colorRGB" in classification and "color" not in classification:
            value = int(classification["colorRGB"]) & 0xFFFFFF
            props["classification"] = {
                **classification,
                "color": [(value >> 16) & 255, (value >> 8) & 255, value & 255],
            }
        for part_index, polygon in enumerate([None] if geometry is None else polygons(geometry)):
            current = dict(props)
            identifier = base_id if part_index == 0 else f"{base_id}-{part_index}"
            if identifier in used:
                identifier += f"-{index}"
            used.add(identifier)
            current["id"] = identifier
            current.pop("simplified", None)
            current.pop("_studio_geometry", None)
            if kind == "tissue":
                # Valid_Tissue is required by classic pixel classes and tissue pieces.
                current["classification"] = {"name": "Valid_Tissue", "color": [46, 204, 113]}
                if not current.get("named"):
                    current["name"] = current["label"] = f"{label}_{len(found) + 1}"
            if kind == "examples":
                role = current.get("example_role") or next((key for key, value in EXAMPLE_CLASSES.items()
                    if (current.get("classification") or {}).get("name") == value["name"]), None)
                if role not in EXAMPLE_CLASSES:
                    raise ValueError("Choose Artifact (red) or Good tissue (green) for this example.")
                current["example_role"] = role
                current["classification"] = dict(EXAMPLE_CLASSES[role])
            # A valid polygon is kept as it was sent (its own coordinates), and checked once: rebuilding and
            # re-checking it took most of the time on large predictions (many small holes).
            same = valid and polygon is geometry and source.get("type") == "Polygon"
            if same:
                if len(_CHECKED) > 50000:
                    _CHECKED.clear()
                _CHECKED[id(source)] = source
            normal = _normal({"type": "Feature", "properties": current,
                              "geometry": source if same else mapping(polygon)}, len(found))
            if normal:
                found.append(normal)
    return found


def _shown(feature):
    """What the browser draws: the polygon itself, or (over the vertex budget) one simplification within half a pixel,
    without holes under 16 pixels² when it has very many (predictions keep thousands of one-pixel gaps that cannot be
    seen). Marked simplified: the file keeps every vertex and a save takes them from it."""
    import shapely
    from shapely.geometry import Polygon, mapping, shape

    props = dict(feature["properties"])
    geometry = shape(feature["geometry"])
    if int(shapely.get_num_coordinates(geometry)) <= shapes.MAX_DISPLAY_VERTICES:
        return {**feature, "properties": props}
    if geometry.geom_type == "Polygon" and len(geometry.interiors) > 100:
        geometry = Polygon(geometry.exterior, [ring for ring in geometry.interiors if Polygon(ring).area >= 16])
    props["simplified"] = True
    return {"type": "Feature", "geometry": mapping(geometry.simplify(0.5, preserve_topology=True)), "properties": props}


_CHECKED = {}  # id(geometry dict) -> that dict: valid single polygons already checked here (never changed in place)


def _normal(feature, index):
    """shapes.normalise for a polygon already made valid here: names, ids and colours, without parsing it again."""
    props = dict(feature.get("properties") or {})
    classification = props.get("classification")
    class_name = classification.get("name") if isinstance(classification, dict) else classification
    color = shapes._hex(classification.get("color")) if isinstance(classification, dict) else None
    color = color or shapes._hex(props.get("color"))
    name = str(props.get("name") or props.get("label") or class_name or f"Shape {index + 1}")
    props.update({"id": str(props.get("id") or uuid.uuid4().hex[:8]), "name": name, "label": name})
    if class_name or color:
        props["classification"] = {"name": str(class_name or name), "color": shapes._rgb(color) or [255, 214, 0]}
    return {"type": "Feature", "geometry": feature["geometry"], "properties": props}


_READ = {}  # (path, kind, label, revision) -> prepared features of that version (a file is read several times per edit)


def read(path, *, kind="regions", label="tissue"):
    path = Path(path)
    raw = path.read_bytes() if path.is_file() else b""
    version = sha256(raw).hexdigest()
    key = (str(path.resolve()), kind, label, version)
    found = _READ.get(key)
    if found is None:
        data = json.loads(raw) if raw else {}
        features = data.get("features", []) if isinstance(data, dict) else data
        found = prepare(features, kind=kind, label=label)
        if len(_READ) > 24:
            _READ.pop(next(iter(_READ)))
        _READ[key] = found
    # Fresh feature and property dicts (callers change them); the geometry is shared and never changed in place.
    return [{**f, "properties": dict(f["properties"])} for f in found], version


_DISPLAY = {}  # id(geometry dict) -> (geometry, its display geometry and simplified flag)


def display_features(features):
    # Tissue can contain thousands of holes. Forcing it below a fixed number
    # of points turns detailed boundaries into triangles even at Full scale.
    # A polygon whose geometry is the same object as before (unchanged since the file was read) reuses its display form.
    out, todo = [None] * len(features), []
    for i, feature in enumerate(features):
        cached = _DISPLAY.get(id(feature["geometry"]))
        if cached and cached[0] is feature["geometry"]:
            props = dict(feature["properties"])
            if cached[2]:
                props["simplified"] = True
            out[i] = {**feature, "geometry": cached[1], "properties": props}
        else:
            todo.append(i)
    if todo:
        made = [_shown(features[i]) for i in todo]
        for i, shown in zip(todo, made):
            out[i] = shown
            if len(_DISPLAY) > 20000:
                _DISPLAY.clear()
            _DISPLAY[id(features[i]["geometry"])] = (features[i]["geometry"], shown["geometry"], bool(shown["properties"].get("simplified")))
    return out


def display(features, version):
    result = display_features(features)
    for feature in result:
        if feature["properties"].get("simplified"):
            feature["properties"]["_studio_geometry"] = version
    return {"features": result, "revision": version}


def _state_path(path, version):
    if not isinstance(version, str) or not re.fullmatch(r"[a-f0-9]{64}", version):
        raise ValueError("Choose a saved version from History.")
    return _home(path) / "states" / f"{version}.json"


def _archive(path, version, features, metadata=None):
    target = _state_path(path, version)
    if not target.exists():
        _atomic(target, json.dumps(features, allow_nan=False).encode())
        _atomic(target.with_suffix(".meta"), json.dumps(metadata or {}).encode())


def _check(path, expected):
    if expected != revision(path):
        raise ValueError(_changed_by(path) + " Reopen the image before editing them.")


def _changed_by(path):
    """What changed a file last, from its History: "P11_roi_1_valid_tissue changed at 12:27 (Detected tissue)."."""
    from datetime import datetime

    events = sorted((_home(path) / "events").glob("*.json"), reverse=True)
    try:
        latest = json.loads(events[0].read_text()) if events else None
    except (OSError, ValueError):
        latest = None
    if not latest:
        return "These annotations changed in another window or job."
    try:
        when = datetime.fromisoformat(latest["when"]).astimezone().strftime("%H:%M")
    except (KeyError, ValueError):
        when = "?"
    return (f"{Path(path).stem} changed at {when} ({latest.get('label', 'an edit')}), in another window, tab or job "
            "(Jobs lists them; Studio servers lists any other SPACEMAP still running).")


def _write(path, features, before, expected, action, *, edited=False, kind="regions", label="tissue", metadata=None, save_id=None):
    clean = prepare(features, kind=kind, label=label)
    old_data = json.loads(Path(path).read_text()) if Path(path).is_file() else {}
    old_meta = old_data.get("studio", {}) if isinstance(old_data, dict) else {}
    derived = {key: old_meta[key] for key in ("source_revision", "source_file", "artifact_revisions", "legacy_cleaned_file", "class_name") if key in old_meta}
    if metadata is not None:
        derived = metadata
    if save_id:  # the page's id for this save, so it can tell its own late save from someone else's change
        derived = {**derived, "save_id": save_id}
    raw = json.dumps(
        {"type": "FeatureCollection", "features": clean, "studio": {**derived, "edited": bool(edited), "action": action}},
        allow_nan=False,
    ).encode()
    after = sha256(raw).hexdigest()
    _archive(path, expected, before, old_meta)
    _archive(path, after, clean, derived)
    now = datetime.now(timezone.utc)
    entry = {
        "id": now.strftime("%Y%m%dT%H%M%S%f") + "-" + uuid.uuid4().hex[:8],
        "before": expected,
        "after": after,
        "label": action,
        "when": now.isoformat(),
        "count": len(clean),
    }
    # Archive first: even an interruption during replacement retains the earlier version.
    _atomic(_home(path) / "events" / (entry["id"] + ".json"), json.dumps(entry).encode())
    _atomic(path, raw)
    _READ[(str(Path(path).resolve()), kind, label, after)] = clean  # the next read of this version needs no parsing
    _follow(path, clean)
    return display(clean, after)


CLASSES_SUFFIX = "_raw_macro_clusters.geojson"


def _follow(path, features):
    """Keep files made from this one in step: the whole-tissue classification's named classes (domains), which Zones,
    Export and Layers read, follow every save, edit, undo and restore of its polygons."""
    name = Path(path).name
    if name.endswith(CLASSES_SUFFIX):
        from .classification import write_domains

        write_domains(Path(path).parent, name[: -len(CLASSES_SUFFIX)], features)


def replace(path, features, expected, action, *, edited=False, kind="regions", label="tissue"):
    with locked(path):
        _check(path, expected)
        before, _ = read(path, kind=kind, label=label)
        return _write(path, features, before, expected, action, edited=edited, kind=kind, label=label)


def last_save_id(path):
    """The page's id of the save that wrote *path* last (none when something else wrote it)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return (data.get("studio") or {}).get("save_id") if isinstance(data, dict) else None


def save(path, edited_features, expected, *, kind="regions", label="tissue", action="Edited by hand", save_id=None, after_save=None):
    """Write the page's polygons. *after_save*: the page's previous save whose reply it never got (a slow server);
    when that save is what changed the file, the file is the version the page expected."""
    if not isinstance(edited_features, list):
        raise ValueError("Send a list of polygons.")
    with locked(path):
        if after_save and expected != revision(path) and last_save_id(path) == after_save:
            expected = revision(path)
        _check(path, expected)
        before, _ = read(path, kind=kind, label=label)
        versions = {expected: {f["properties"]["id"]: f for f in before}}
        merged = []
        for feature in edited_features:
            props = dict(feature.get("properties") or {})
            if props.get("simplified"):
                version = props.get("_studio_geometry", expected)
                if version not in versions:
                    stored = _state_path(path, version)
                    if not stored.is_file():
                        raise ValueError("The detailed outline is no longer available. Reopen the image.")
                    versions[version] = {f["properties"]["id"]: f for f in json.loads(stored.read_text())}
                original = versions[version].get(str(props.get("id")))
                if original is None:
                    raise ValueError("The detailed outline changed. Reopen the image.")
                feature = {**feature, "geometry": original["geometry"]}
            merged.append(feature)
        return _write(path, merged, before, expected, action, edited=True, kind=kind, label=label, save_id=save_id)


def history(path):
    entries = []
    for item in sorted((_home(path) / "events").glob("*.json"), reverse=True):
        entry = json.loads(item.read_text())
        entries.append(entry)
    return {"entries": entries, "revision": revision(path)}


def restore(path, version, expected, *, kind="regions", label="tissue"):
    snapshot = _state_path(path, version)
    if not snapshot.is_file():
        raise ValueError("That version is no longer available. Reload History.")
    metadata_path = snapshot.with_suffix(".meta")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    with locked(path):
        _check(path, expected)
        before, _ = read(path, kind=kind, label=label)
        return _write(path, json.loads(snapshot.read_text()), before, expected, "Restored an earlier version",
                      edited=True, kind=kind, label=label, metadata=metadata)


def brush_geometry(geometry, points, radius, mode, bounds=None):
    """Apply one vector stroke at full precision. Disconnected additions are refused."""
    from shapely.geometry import LineString, Point, box, mapping, shape
    from shapely.ops import unary_union

    if mode not in ("add", "erase"):
        raise ValueError("Choose paint or erase.")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(radius)
        or not 0 < radius <= 100000
    ):
        raise ValueError("Choose a brush size greater than zero.")
    if not isinstance(points, list) or not 1 <= len(points) <= 10000:
        raise ValueError("Draw a shorter brush stroke.")
    values = shapes._points(points, 1, "The brush stroke")
    stroke = (LineString(values) if len(set(values)) > 1 else Point(values[0])).buffer(radius, resolution=16)
    if bounds:
        stroke = stroke.intersection(box(0, 0, *bounds))
    original = shape(geometry)
    if mode == "add" and not stroke.intersects(original):
        raise ValueError("Start the brush on the selected polygon or its edge.")
    result = original.union(stroke) if mode == "add" else original.difference(stroke)
    result = unary_union(polygons(result))
    return [mapping(p) for p in sorted(polygons(result), key=lambda p: (-p.area, p.bounds))]


def brush(path, identifier, points, radius, mode, expected, *, kind="regions", label="tissue", bounds=None):
    with locked(path):
        _check(path, expected)
        features, _ = read(path, kind=kind, label=label)
        target = next((f for f in features if f["properties"]["id"] == identifier), None)
        if target is None:
            raise ValueError("Select a polygon first.")
        parts = brush_geometry(target["geometry"], points, radius, mode, bounds)
        made = []
        for index, geometry in enumerate(parts):
            props = dict(target["properties"])
            if index:
                props["id"] = uuid.uuid4().hex[:20]
            made.append({"type": "Feature", "properties": props, "geometry": geometry})
        updated = [part for f in features for part in (made if f is target else [f])]
        result = _write(
            path,
            updated,
            features,
            expected,
            "Painted tissue"
            if mode == "add" and kind == "tissue"
            else "Painted polygon"
            if mode == "add"
            else "Erased with brush",
            edited=True,
            kind=kind,
            label=label,
        )
        result["selected"] = [f["properties"]["id"] for f in made]
        return result


def edit_geometry(geometries, operation, *, line=None, width=2, gap=3):
    """Exact split/union, with only a narrow cut or explicit gap bridges added.

    Unlike the legacy cell-outline tools, this does not round outlines or fill
    holes. Distances are full-resolution image pixels, independent of scale.
    """
    from shapely.geometry import LineString, Point, mapping, shape
    from shapely.ops import nearest_points, unary_union
    from shapely.strtree import STRtree

    if operation not in ("split", "merge"):
        raise ValueError("Choose Split or Merge.")
    value = width if operation == "split" else gap
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 100000
            or (operation == "split" and value == 0)):
        raise ValueError("Use a positive cut width or a maximum gap of zero or more, in pixels.")
    if not isinstance(geometries, list) or not geometries:
        raise ValueError("Select polygons first.")
    originals = [shape(item) for item in geometries]
    if any(p.is_empty or not p.is_valid or p.geom_type not in ("Polygon", "MultiPolygon") for p in originals):
        raise ValueError("Select valid polygons first.")
    if operation == "split":
        if len(originals) != 1:
            raise ValueError("Select one polygon to split.")
        points = shapes._points(line, 2, "The cutting line")
        blade = LineString(points)
        if blade.length < 2:
            raise ValueError("Drag a longer line across the polygon.")
        result = originals[0].difference(blade.buffer(width / 2, cap_style=2))
        if len(polygons(result)) <= len(polygons(originals[0])):
            raise ValueError("Draw right across the neck, from outside one edge to outside the other.")
    else:
        if len(originals) < 2:
            raise ValueError("Select at least two polygons to merge.")
        result = unary_union(originals)
        parts = polygons(result)
        # Join the nearest components, adding a 2 px wide bridge only between
        # those within the requested gap. Never smooth all the original edges.
        if len(parts) > 1:
            tree = STRtree(parts)
            edges = []
            for i, part in enumerate(parts):
                for j in tree.query(part.buffer(gap + 1e-9).envelope):
                    if j < i:
                        distance = part.distance(parts[j])
                        if distance <= gap:
                            edges.append((distance, i, int(j)))
            parents = list(range(len(parts)))
            def parent(i):
                while parents[i] != i:
                    parents[i] = parents[parents[i]]
                    i = parents[i]
                return i
            bridges = []
            for distance, a, b in sorted(edges):
                pa, pb = parent(a), parent(b)
                if pa == pb:
                    continue
                p, q = nearest_points(parts[a], parts[b])
                bridges.append((LineString([p, q]) if distance > 0 else Point(p)).buffer(1))
                parents[pa] = pb
            if len({parent(i) for i in range(len(parts))}) > 1:
                raise ValueError("These polygons are too far apart. Increase Maximum gap to join them.")
            result = unary_union([*parts, *bridges])
    return [mapping(p) for p in sorted(polygons(result), key=lambda p: (-p.area, p.bounds))]


def edit(path, identifiers, operation, expected, *, line=None, width=2, gap=3, kind="regions", label="tissue"):
    """Split or merge the full saved outlines under the same revision lock as brush."""
    if not isinstance(identifiers, list) or not identifiers or any(not isinstance(i, str) for i in identifiers):
        raise ValueError("Select polygons first.")
    with locked(path):
        _check(path, expected)
        before, _ = read(path, kind=kind, label=label)
        selected = set(identifiers)
        targets = [f for f in before if f["properties"]["id"] in selected]
        if len(targets) != len(selected):
            raise ValueError("These polygons changed. Select them again.")
        if kind == "examples" and operation == "merge" and len({f["properties"]["example_role"] for f in targets}) > 1:
            raise ValueError("Merge red examples together or green examples together; keep the two labels separate.")
        parts = edit_geometry([f["geometry"] for f in targets], operation, line=line, width=width, gap=gap)
        made = [{"type": "Feature", "geometry": g,
                 "properties": {**targets[0]["properties"], "id": uuid.uuid4().hex[:20] if i else targets[0]["properties"]["id"]}}
                for i, g in enumerate(parts)]
        updated = [part for f in before for part in (made if f is targets[0] else [] if f in targets else [f])]
        result = _write(path, updated, before, expected, "Split polygon" if operation == "split" else "Merged polygons",
                        edited=True, kind=kind, label=label)
        result["selected"] = [f["properties"]["id"] for f in made]
        return result


def neck_features(features, cutout, ids=None):
    """Apply one reviewed neck footprint, keeping each tissue piece's identity."""
    from shapely.geometry import mapping, shape

    blade = shape(cutout)
    if blade.is_empty or not blade.is_valid or blade.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("This neck has no valid cut area. Find necks again.")
    made, selected = [], []
    wanted = set(ids) if ids is not None else None
    changed = False
    for feature in features:
        original = shape(feature["geometry"])
        if (wanted is not None and feature['properties']['id'] not in wanted) or not original.intersects(blade):
            made.append(feature)
            continue
        parts = sorted(polygons(original.difference(blade)), key=lambda p: -p.area)
        if len(parts) <= len(polygons(original)):
            raise ValueError("This suggested cut does not separate the tissue. Use Split or the erase brush to adjust it.")
        changed = True
        for i, part in enumerate(parts):
            props = dict(feature["properties"])
            if i:
                props["id"] = uuid.uuid4().hex[:20]
            props.pop("simplified", None)
            props.pop("_studio_geometry", None)
            selected.append(props["id"])
            made.append({"type":"Feature", "properties":props, "geometry":mapping(part)})
    if not changed:
        raise ValueError("This neck no longer touches the tissue. Find necks again.")
    return {"features":made, "selected":selected}


def approve_neck(path, cutout, expected, *, kind="tissue", label="tissue", ids=None):
    with locked(path):
        _check(path, expected)
        before, _ = read(path, kind=kind, label=label)
        edited = neck_features(before, cutout, ids)
        result = _write(path, edited["features"], before, expected, "Approved neck cut", edited=True, kind=kind, label=label)
        result["selected"] = edited["selected"]
        return result


def subtract(tissue, artifact_paths, expected, *, label="tissue", target=None, target_revision=None, overrides=None,
             kind='tissue', class_name='', artifact_revisions=None):
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    from contextlib import ExitStack
    with ExitStack() as stack:
        for path in sorted({Path(tissue), *(Path(p) for p in artifact_paths)}):
            stack.enter_context(locked(path))
        _check(tissue, expected)
        features, _ = read(tissue, kind=kind, label=label)
        if not features:
            raise ValueError("Detect tissue first.")
        artifacts = []
        for path in artifact_paths:
            if artifact_revisions is not None: _check(path, artifact_revisions[str(path)])
            for feature in read(path, kind="artifacts")[0]:
                props = feature.get("properties") or {}
                classification = props.get("classification") or {}
                name = classification.get("name") if isinstance(classification, dict) else classification
                if props.get("source") == "auto_artifact" and name in (overrides or {}).get(str(path), ()):
                    continue
                artifacts.append(feature)
        if not artifacts and (target is None or not any(Path(p).is_file() for p in artifact_paths)):
            raise ValueError("Detect or draw artifacts first.")
        cut = unary_union([shape(f["geometry"]) for f in artifacts])
        from .annotation_inputs import annotation_class
        if class_name and not any(annotation_class(f) == class_name for f in features):
            raise ValueError('The chosen annotation class no longer exists. Choose Work within again.')
        result = [{**f, "geometry": mapping(shape(f["geometry"]).difference(cut))}
                  if not class_name or annotation_class(f) == class_name else f for f in features]
        if target is None:
            return _write(tissue, result, features, expected, "Subtracted artifacts", kind=kind, label=label,
                          metadata={'artifact_revisions':artifact_revisions or {}, 'class_name':class_name})
        if Path(target).resolve() == Path(tissue).resolve():
            raise ValueError("Cleaned tissue needs a separate file.")
        with locked(target):
            _check(target, target_revision)
            previous, _ = read(target, kind="tissue", label=label)
            return _write(target, result, previous, target_revision, "Subtracted artifacts", kind="tissue", label=label,
                          metadata={"source_revision": expected, "source_file": str(Path(tissue).resolve())})


def shown(layer):
    """A registered shapes layer's polygons as the browser shows them (an annotation layer comes as pieces with their
    ids), so ids picked on the image are found again."""
    if layer.get("annotation_kind"):
        features, _ = read(layer["path"], kind=layer["annotation_kind"], label=layer.get("tissue_label", "tissue"))
    else:
        features, _ = read(layer["path"], kind="shapes")
    return [f for f in features if (f.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")]
