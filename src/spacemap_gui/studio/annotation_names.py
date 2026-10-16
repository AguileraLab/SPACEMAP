"""An image's annotations by name, as Layers shows them: which polygons are exported, and the export itself.

Every polygon belongs to one name, by the same rule as Layers (viewer/regions.js, annotationName): a name given by
hand wins; tissue pieces (tissue_1, tissue_2…) belong to the tissue; artifact polygons to their artifact (Blood
clots…), as Artifact or Good tissue (training examples); otherwise the polygon's class. A name is keyed
``<source>::<name>::<role>`` (source empty for the image's own layers, the file's name for an imported one).

Every name is *local* (made in Studio, or a file in the project's outputs) or *imported* (a GeoJSON brought in, or a
file found outside the outputs). Export follows one set of rules for the whole project, keyed ``<origin>:<name>``
(``local:tissue``, ``imported:tumor``): Export (write it), Subtract (cut it out of the others), an optional export
name and colour; and Keep inside (every exported polygon is cut to that annotation). Each image applies the rules to
the annotations it has; a name it lacks is skipped. The source files are never changed by an export.
The names given by hand are also kept for the whole project with their colours, so every image offers them.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

AUTOMATIC = re.compile(r"^(Valid_Tissue|Marker_Region|Region \d+|Unnamed|Artifact|Good tissue|Sharp tissue|Good glass)$")
ROLE_SUFFIX = {"good": " · Good tissue"}


def _class(properties):
    value = properties.get("classification")
    return str(value.get("name") or "") if isinstance(value, dict) else str(value or "")


def name_of(feature, kind, layer_name):
    """The name row a polygon belongs to (the same rule as Layers)."""
    properties = feature.get("properties") or {}
    if kind in ("artifact", "examples"):
        return layer_name
    if properties.get("named") and properties.get("name"):
        return str(properties["name"])
    if kind == "tissue":
        return layer_name
    cls = _class(properties)
    if cls and not AUTOMATIC.match(cls):
        return cls
    if re.fullmatch(r"Cluster \d+", str(properties.get("name") or "")):
        return str(properties["name"])
    return re.sub(r"^Marker regions · ", "", layer_name or "Annotations")


def files(root, config, image, output):
    """The image's annotation files: (path, kind, layer name, source, read only)."""
    from . import annotation, annotation_inputs, annotation_layers, artifact_learning as learning

    sample = annotation.sample_id(image)
    found, seen = [], set()
    outputs = (Path(root) / "outputs").resolve()

    def add(path, kind, name, source="", readonly=False):
        path = Path(path)
        key = str(path.resolve())
        if path.is_file() and key not in seen:
            seen.add(key)
            outside = kind == "imported" or kind == "file" and outputs not in path.resolve().parents
            found.append({"path": key, "kind": kind, "layer": name, "source": source, "readonly": readonly,
                          "origin": "imported" if outside else "local"})

    add(annotation.valid_tissue(output, sample), "tissue", annotation.tissue_label(config))
    for part, name in learning.NAMES.items():
        add(learning.polygons_path(output, sample, part), "artifact", name)
        add(learning.example_path(root, part, image), "examples", name)
    add(annotation.drawn_artifacts(output, sample), "drawn", "Artifacts")
    for path, info in annotation_layers.generated(root, image, output).items():
        add(path, "regions", info["name"])
    for path in annotation_inputs.imported(root, image):
        stem = Path(path).stem.split("_", 1)[-1]
        add(path, "imported", stem, source=stem)
    cleaned = str(annotation.cleaned_tissue(output, sample).resolve())
    for path, label in annotation_inputs.sources(root, config, image, output).items():
        if path != cleaned and "studio_export" not in Path(path).parts and not studio_export(path):
            add(path, "file", label, source=label, readonly=True)  # read as it is (zones, classic outputs)
    return found


_EXPORTS = {}


def studio_export(path):
    """Whether a GeoJSON is a file Export wrote (its polygons carry their source layer and feature): an output, never
    read back as annotations of the image (it would add them again as local names). Only the file's start is read
    (Export writes those properties in every polygon), and the answer is kept until the file changes."""
    from . import file_facts

    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return False
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _EXPORTS:
        kept = file_facts.get(path, "studio_export")
        if kept is None:
            try:
                with open(path, "rb") as handle:
                    head = handle.read(262144)
                kept = b'"source_feature_id"' in head and b'"source_layer"' in head
            except OSError:
                kept = False
            file_facts.put(path, "studio_export", kept)
        _EXPORTS[key] = bool(kept)
        if len(_EXPORTS) > 2048:
            _EXPORTS.clear()
    return _EXPORTS[key]


def choices(root, config, image, output):
    """Layers' Export ticks for the open image: each name row ticked when its rule exports it."""
    return {g["key"]: g["export"] for g in groups(root, config, image, output)}


def save_choices(root, config, image, output, changes):
    """Tick or untick name rows ({key: true/false}) in Layers: the rule of each (for every image) changes."""
    if not isinstance(changes, dict) or not changes or len(changes) > 2000 or not all(
        isinstance(k, str) and len(k) <= 600 and isinstance(v, bool) for k, v in changes.items()
    ):
        raise ValueError("Tick names to export.")
    known = {g["key"]: g["rule"] for g in groups(root, config, image, output)}
    rules, unsaved = {}, {}
    for key, value in changes.items():
        rule = known.get(key)
        if rule is None:  # a name given in Layers whose polygons are not saved yet: its rule from the row key
            source, _, rest = key.partition("::")
            name, _, role = rest.rpartition("::")
            rule = "" if role == "good" or not name else ("imported:" if source else "local:") + name
            unsaved[key] = value
        if rule:
            rules[rule] = {"export": value}
    if rules:
        from .settings import read_config

        save_export_settings(root, {"rules": rules})
        config = read_config(root)
    return {**choices(root, config, image, output), **unsaved}


def _read(entry, label):
    from . import annotation_edits as edits

    kind = "tissue" if entry["kind"] == "tissue" else "regions"
    return edits.read(entry["path"], kind=kind, label=label)


_SCANS = {}


def _scan(entry):
    """The names (and colours) in an annotation file, read quickly: no geometry checks, a polygon's parts counted.
    Kept until the file changes. Listing names must stay fast on files with hundreds of thousands of polygons; the
    full, checked read is only needed to export."""
    path = Path(entry["path"])
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (str(path), stat.st_mtime_ns, stat.st_size, entry["kind"], entry["layer"])
    if key in _SCANS:
        return _SCANS[key]
    from . import file_facts

    kept = file_facts.get(path, f"names:{entry['kind']}:{entry['layer']}")
    if kept is not None:
        _SCANS[key] = [tuple(item) for item in kept]
        return _SCANS[key]
    try:
        data = json.loads(path.read_bytes() or b"{}")
    except (OSError, ValueError):
        return []
    features = data.get("features", []) if isinstance(data, dict) else data
    found = []
    for feature in features or []:
        geometry = (feature or {}).get("geometry") or {}
        kind = geometry.get("type")
        parts = len(geometry.get("coordinates") or []) if kind == "MultiPolygon" else 1 if kind == "Polygon" else 0
        if not parts:
            continue
        properties = feature.get("properties") or {}
        cls = properties.get("classification")
        colour = properties.get("colour") if isinstance(properties.get("colour"), str) else None
        if colour is None and isinstance(cls, dict) and isinstance(cls.get("color"), list) and len(cls["color"]) >= 3:
            colour = "#" + "".join(f"{int(v):02x}" for v in cls["color"][:3])
        found.append((name_of(feature, entry["kind"], entry["layer"]), colour, parts))
    if len(_SCANS) > 512:
        _SCANS.clear()
    _SCANS[key] = found
    file_facts.put(path, f"names:{entry['kind']}:{entry['layer']}", [list(item) for item in found])
    return found


def groups(root, config, image, output):
    """Every name with its polygons count, colour, files, origin, rule (``<origin>:<name>``; none for training
    examples) and whether its rule exports it."""
    rules = export_settings(root, config)["rules"]
    found = {}
    for entry in files(root, config, image, output):
        role = "good" if entry["kind"] == "examples" else "artifact" if entry["kind"] == "artifact" else ""
        for name, colour, parts in _scan(entry):
            key = f"{entry['source']}::{name}::{role}"
            group = found.setdefault(key, {"key": key, "source": entry["source"], "name": name, "role": role, "count": 0,
                                           "files": [], "colour": None, "readonly": entry["readonly"], "origin": entry["origin"],
                                           "rule": "" if role == "good" else f"{entry['origin']}:{name}"})
            group["count"] += parts
            if entry["path"] not in group["files"]:
                group["files"].append(entry["path"])
            if group["colour"] is None and colour:
                group["colour"] = colour
    for group in found.values():
        group["export"] = bool(group["rule"] and (rules.get(group["rule"]) or {}).get("export"))
    return sorted(found.values(), key=lambda g: (g["source"], g["name"].lower(), g["role"]))


def _near(whole):
    """For a large shape (a slide's tissue, its artifacts), a function giving only its part around one polygon (None
    when they do not meet). Cutting a polygon against the nearby part gives the same result as against the whole,
    without walking the whole slide's outline for every polygon."""
    import shapely
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
    from shapely.validation import make_valid

    from .annotation_edits import polygons

    if whole is None:
        return lambda geometry: None
    parts = polygons(whole)
    tree = STRtree(parts)

    def near(geometry):
        hits = tree.query(geometry, predicate="intersects")
        if not len(hits):
            return None
        x0, y0, x1, y1 = geometry.bounds
        window = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
        # A fast clip to the polygon's box can leave a shape slightly invalid: repaired, else cut exactly.
        clipped = [make_valid(shapely.clip_by_rect(parts[i], *window)) for i in hits]
        try:
            local = make_valid(unary_union([c for c in clipped if not c.is_empty]))
        except shapely.errors.GEOSException:
            local = make_valid(unary_union([parts[i] for i in hits])).intersection(shapely.box(*window))
        return None if local.is_empty else local
    return near


def rule_label(rule):
    """``imported:tissue`` as "tissue (imported)"."""
    origin, _, name = str(rule).partition(":")
    return f"{name} ({origin})" if name else str(rule)


EXPORT_KEY = "annot_studio_export"
# folder "": the Annotation output folder; inside "": no cutting, else the rule every exported polygon is cut to;
# rules: {"<origin>:<name>": {"export": bool, "subtract": bool, "name": export name, "colour": "#rrggbb"}}
# smallest_um2: pieces of Keep inside (after Subtract) smaller than this are dropped with everything in them; 0: kept
EXPORT_DEFAULTS = {"prefix": "", "suffix": "", "folder": "", "merge": True, "inside": "", "rules": {}, "smallest_um2": 0}
RULE = re.compile(r"(local|imported):.{1,120}")
AFFIX = re.compile(r"[A-Za-z0-9_.+-]{0,40}")


def export_settings(root, config):
    """How exported files are named and where they go: <prefix><image name><suffix>.geojson in the folder."""
    saved = next((tab[EXPORT_KEY] for tab in (config.get("tab_configurations") or {}).values()
                  if isinstance(tab, dict) and isinstance(tab.get(EXPORT_KEY), dict)), {})
    values = {**EXPORT_DEFAULTS, **{k: v for k, v in saved.items() if k in EXPORT_DEFAULTS}}
    if not RULE.fullmatch(str(values["inside"] or "")):
        values["inside"] = ""  # an older choice (by file) is not a rule
    values["rules"] = values["rules"] if isinstance(values["rules"], dict) else {}
    return values


def export_folder(root, settings, config=None):
    """Where exported files go: the Annotation step's Output folder itself (its working files are in the
    Sub-directory). One folder, chosen at the top of the step."""
    from . import annotation
    from .settings import read_config

    config = config if config is not None else read_config(root)
    return annotation.main_output(root, config.get("directory_mappings") or {})


def file_name(image, settings):
    from . import annotation

    return f"{settings['prefix']}{annotation.sample_id(Path(image).name)}{settings['suffix']}.geojson"


def save_export_settings(root, changes):
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if not isinstance(changes, dict) or set(changes) - set(EXPORT_DEFAULTS):
        raise ValueError("Only the export's prefix, suffix, folder, merging, Keep inside and rules can be changed.")
    for key in ("prefix", "suffix"):
        if key in changes and not (isinstance(changes[key], str) and AFFIX.fullmatch(changes[key])):
            raise ValueError("A prefix or suffix uses letters, numbers and _ . + - (up to 40), no spaces.")
    if "folder" in changes and not (isinstance(changes["folder"], str) and len(changes["folder"]) <= 1024):
        raise ValueError("Type the export folder's path.")
    if "merge" in changes and not isinstance(changes["merge"], bool):
        raise ValueError("Merge same names is on or off.")
    if "smallest_um2" in changes and not (isinstance(changes["smallest_um2"], (int, float)) and not isinstance(changes["smallest_um2"], bool)
                                          and 0 <= changes["smallest_um2"] <= 1e9):
        raise ValueError("Give the smallest piece to keep in µm² (0 keeps every piece).")
    if "inside" in changes and not (changes["inside"] == "" or isinstance(changes["inside"], str) and RULE.fullmatch(changes["inside"])):
        raise ValueError("Choose the annotation to keep the export inside.")
    rules = changes.get("rules")
    if rules is not None:
        if not isinstance(rules, dict) or len(rules) > 2000:
            raise ValueError("Give the export rules.")
        for rule, value in rules.items():
            if not (isinstance(rule, str) and RULE.fullmatch(rule) and isinstance(value, dict) and set(value) <= {"export", "subtract", "name", "colour"}):
                raise ValueError("Give the export rules.")
            if any(k in value and not isinstance(value[k], bool) for k in ("export", "subtract")):
                raise ValueError("Export and Subtract are on or off.")
            if value.get("name") and not NAME.fullmatch(str(value["name"])):
                raise ValueError("Names use letters, numbers, spaces and _ . + ( ) - (up to 60).")
            if value.get("colour") and not COLOUR.fullmatch(str(value["colour"])):
                raise ValueError("Choose a colour.")
    with project_lock(root):
        config = read_config(root)
        current = export_settings(root, config)
        values = {**current, **changes}
        if rules is not None:  # rules change one at a time; the others stay
            values["rules"] = {**current["rules"], **{k: {**current["rules"].get(k, {}), **v} for k, v in rules.items()}}
        if not export_folder(root, values).parent.is_dir():
            raise ValueError(f"{export_folder(root, values).parent} is not a folder.")
        persistence.update_stored_config_key(config, EXPORT_KEY, values)
        persistence.save_project_config(str(root), config)
    return values


def export(root, config, image, output, merge=True, naming=None):
    """One GeoJSON of this image by the export rules (*naming*, else the saved ones): the annotations ticked Export,
    cut to Keep inside, with those ticked Subtract cut out, each named and coloured as its rule says; same names
    merged. It is kept where the later steps read it (the Annotation output's studio_export) and, with *naming*,
    also written as <prefix><image name><suffix>.geojson in the export folder. Names this image lacks are skipped."""
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    import shapely

    from . import annotation, annotation_edits as edits
    from .annotation_layers import merge_same_names
    from .settings import project_lock

    label = annotation.tissue_label(config)
    settings = naming if naming is not None else export_settings(root, config)
    rules, inside = settings.get("rules") or {}, str(settings.get("inside") or "")
    rule_of = lambda r: rules.get(r) or {}
    with project_lock(root):
        present = [g for g in groups(root, config, image, output) if g["rule"]]
        chosen = [g for g in present if rule_of(g["rule"]).get("export")]
        if not chosen:
            raise ValueError("None of the annotations ticked Export is on this image.")
        container = [g for g in present if g["rule"] == inside] if inside else []
        if inside and not container:
            raise ValueError(f"No {rule_label(inside)} on this image to keep the export inside.")
        taken = [g for g in present if rule_of(g["rule"]).get("subtract")]
        by_key = {g["key"]: g for g in present}
        needed = {path for g in chosen + container + taken for path in g["files"]}  # only these files are read
        exported, area, minus, holders, sources = [], [], [], set(), []
        for entry in files(root, config, image, output):
            if entry["path"] not in needed:
                continue
            saved, version = _read(entry, label)
            role = "good" if entry["kind"] == "examples" else "artifact" if entry["kind"] == "artifact" else ""
            used = 0
            for feature in saved:
                name = name_of(feature, entry["kind"], entry["layer"])
                group = by_key.get(f"{entry['source']}::{name}::{role}")
                if group is None:
                    continue
                rule = group["rule"]
                if rule == inside:
                    area.append(make_valid(shape(feature["geometry"])))
                    if entry["kind"] == "tissue":  # the tissue layer: its pieces renamed by hand are not cut away
                        holders.add(entry["path"])
                if rule_of(rule).get("subtract"):
                    minus.append(make_valid(shape(feature["geometry"])))
                if not rule_of(rule).get("export"):
                    continue
                item = deepcopy(feature)
                properties = item["properties"]
                final = rule_of(rule).get("name") or name
                cls = properties.get("classification") if isinstance(properties.get("classification"), dict) else {}
                colour = rule_of(rule).get("colour") or colour_of(properties)
                if colour:
                    cls = {**cls, "color": [int(colour[i:i + 2], 16) for i in (1, 3, 5)]}
                identifier = properties.get("id", "")
                properties.update(objectType="annotation", name=final, label=final, classification={**cls, "name": final}, source_layer=entry["path"],
                                  source_layer_revision=version, source_feature_id=identifier, export_rule=rule,
                                  id=sha256((entry["path"] + "|" + str(identifier)).encode()).hexdigest()[:24])
                exported.append(item)
                used += 1
            if used:
                sources.append({"path": entry["path"], "layer": entry["layer"], "revision": version, "polygons": used})
        area = make_valid(unary_union(area)) if area else None
        minus = make_valid(unary_union(minus)) if minus else None
        # Pieces of Keep inside (once Subtract is cut out) smaller than the cutoff go, with everything in them.
        smallest, dropped = float(settings.get("smallest_um2") or 0), 0
        if area is not None and smallest > 0:
            from spacemap.viewer.sources import image_info

            microns = image_info(str(image)).pixel_size_um or float(annotation.settings(config).get("microns_per_pixel", 0.5083))
            pieces = edits.polygons(area.difference(minus) if minus is not None else area)
            kept = [piece for piece in pieces if piece.area * microns ** 2 >= smallest]
            dropped = len(pieces) - len(kept)
            if not kept:
                raise ValueError(f"Every piece of {rule_label(inside)} is smaller than {smallest:g} µm².")
            if dropped:
                area = make_valid(unary_union(kept))
        features = []
        for value in (area, minus):
            if value is not None:
                shapely.prepare(value)
        near_area, near_minus = _near(area), _near(minus)
        for item in exported:
            properties = item["properties"]
            rule = properties["export_rule"]
            geometry, changed = make_valid(shape(item["geometry"])), False
            # Pieces renamed by hand inside the Keep inside annotation's own layer (a lymph node in the tissue) stay.
            cut_to = area is not None and (rule != inside and properties["source_layer"] not in holders or rule == inside and dropped)
            if cut_to and not area.contains_properly(geometry):
                local = near_area(geometry)
                if local is None:
                    continue
                geometry, changed = geometry.intersection(local), True
            if minus is not None and not rule_of(rule).get("subtract"):
                local = near_minus(geometry)
                if local is not None:
                    geometry, changed = geometry.difference(local), True
            if changed:
                pieces = [piece for piece in edits.polygons(geometry) if piece.area >= 1]
                if not pieces:
                    continue
                item["geometry"] = mapping(pieces[0] if len(pieces) == 1 else unary_union(pieces))
            features.append(item)
        # Ticked annotations the cutting left nothing of: said, so a wrong Keep inside is seen at once.
        kept_rules = {f["properties"]["export_rule"] for f in features}
        left_out = sorted({rule_label(g["rule"]) for g in chosen if g["rule"] not in kept_rules})
        if not features:
            raise ValueError("Nothing is left to export once Keep inside and Subtract are applied" +
                             (f" ({', '.join(left_out)} lie outside {rule_label(inside)})." if inside and left_out else "."))
        if merge:
            features = merge_same_names(features)
        features = [f for f in features if shape(f["geometry"]).area >= 1]  # no empty slivers
        # Written as it is, one feature per name when merged (as QuPath writes it): an export is an output, not a layer
        # to edit, so it is not split into pieces and keeps no History copies.
        main = annotation.main_output(root, config.get("directory_mappings") or {})
        target = annotation.exported(main, annotation.sample_id(image))
        raw = json.dumps({"type": "FeatureCollection", "features": features}, allow_nan=False, separators=(",", ":")).encode()
        edits._atomic(target, raw)
        revision = sha256(raw).hexdigest()
        names = sorted({f["properties"].get("name") for f in features})
        filename = target.name
        if naming:
            named = export_folder(root, naming, config) / file_name(image, naming)
            named.parent.mkdir(parents=True, exist_ok=True)
            edits._atomic(named, raw)
            if named.parent.resolve() == main.resolve():  # remembered, so a move to the Sub-directory leaves it here
                listed = main / "studio_export" / "named.json"
                try:
                    known = set(json.loads(listed.read_text()))
                except (OSError, ValueError, TypeError):
                    known = set()
                edits._atomic(listed, json.dumps(sorted(known | {named.name})).encode())
            filename = named.name
        return {"path": str(target), "filename": filename, "saved_as": str(named) if naming else "", "count": len(features), "names": names,
                "inside": rule_label(inside) if inside else "", "subtracted": sorted({rule_label(g["rule"]) for g in taken}), "left_out": left_out,
                "small_pieces_dropped": dropped,
                "layers": len(sources), "sources": sources, "revision": revision,
                "geojson": {"type": "FeatureCollection", "features": features}}


def rename(root, config, image, output, key, name=None, colour=None):
    """Rename and/or recolour every polygon of a name row (``<source>::<name>::<role>``) in its files, with History.
    A file read as it is (classic outputs, a project GeoJSON) is never changed: its polygons are first imported as an
    editable copy, which takes the file's place in this image's list. Returns the files changed and the new copies."""
    from . import annotation, annotation_edits as edits, annotation_inputs
    from .settings import project_lock

    source, _, rest = str(key).partition("::")
    old, _, role = rest.rpartition("::")
    new = str(name).strip() if name is not None else None
    if new is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_ .+()-]{0,59}", new):
        raise ValueError("Names use letters, numbers, spaces and _ . + ( ) - (up to 60).")
    if colour is not None and not (isinstance(colour, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", colour)):
        raise ValueError("Choose a colour.")
    label = annotation.tissue_label(config)
    changed, copies = [], []
    with project_lock(root):
        entries = [e for e in files(root, config, image, output) if e["source"] == source and
                   ("good" if e["kind"] == "examples" else "artifact" if e["kind"] == "artifact" else "") == role]
        for entry in entries:
            features, version = _read(entry, label)
            hits = [f for f in features if name_of(f, entry["kind"], entry["layer"]) == old]
            if not hits:
                continue
            if new is not None and new != old and (entry["kind"] in ("artifact", "examples") or
                                                   entry["kind"] == "tissue" and not all(f["properties"].get("named") for f in hits)):
                raise ValueError(f"{old} is named by its step (Tissue or Artifacts): change it there, or Duplicate it under a new name.")
            path = entry["path"]
            if entry["readonly"]:  # an editable copy takes the file's place
                path = str(annotation_inputs.import_polygons(root, image, Path(entry["path"]).name, features))
                hide_file(root, image, entry["path"])
                copies.append(path)
                features, version = edits.read(path)
                hits = [f for f in features if name_of(f, "imported", Path(path).stem.split("_", 1)[-1]) == old]
            for feature in hits:
                properties = feature["properties"]
                cls = properties.get("classification") if isinstance(properties.get("classification"), dict) else {}
                shown = colour or colour_of(properties)
                if new is not None and new != old:
                    properties.update(name=new, label=new, named=True)
                    cls = {**cls, "name": new}
                if colour is not None:
                    properties["colour"] = colour
                    cls = {**cls, "color": [int(colour[i:i + 2], 16) for i in (1, 3, 5)]}
                properties["classification"] = cls
            kind = "tissue" if entry["kind"] == "tissue" else "regions"
            edits.replace(path, features, version, f"Renamed {old} to {new}" if new and new != old else f"Recoloured {old}",
                          edited=True, kind=kind, label=label)
            changed.append(path)
        if not changed:
            raise ValueError(f"{old} has no polygons here any more. Refresh the list.")
        if not role and shown and (entries[0]["kind"] != "tissue" or new not in (None, old)):  # offered on every image from now on
            remember(root, {new if new is not None else old: shown})
        if new is not None and new != old and role != "good":  # the rule goes with the new name
            origin = "imported" if copies or entries[0]["origin"] == "imported" else "local"
            rules = export_settings(root, config)["rules"]
            if f"{entries[0]['origin']}:{old}" in rules and f"{origin}:{new}" not in rules:
                save_export_settings(root, {"rules": {f"{origin}:{new}": rules[f"{entries[0]['origin']}:{old}"]}})
    return {"changed": changed, "copies": copies}


# ---------------------------------------------------------------- the project's names
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_ .+()-]{0,59}")
COLOUR = re.compile(r"#[0-9a-fA-F]{6}")


def colour_of(properties):
    """A polygon's colour as #rrggbb: the colour chosen for its name, else its class colour."""
    chosen = properties.get("colour")
    if isinstance(chosen, str) and COLOUR.fullmatch(chosen):
        return chosen.lower()
    cls = properties.get("classification")
    value = cls.get("color") if isinstance(cls, dict) else None
    if isinstance(value, list) and len(value) >= 3 and all(isinstance(v, (int, float)) for v in value[:3]):
        return "#" + "".join(f"{max(0, min(255, int(v))):02x}" for v in value[:3])
    return None


def _palette_path(root):
    return Path(root) / "outputs/studio/annotation_names.json"


def palette(root):
    """The names given by hand in this project, most recently used first, each with its colour: every image offers
    them (right click on polygons, Label selected polygons), so Normal and Tumor are named once."""
    try:
        value = json.loads(_palette_path(root).read_text())
    except (OSError, ValueError):
        return []
    items = value.get("names") if isinstance(value, dict) else None
    return [{"name": i["name"], "colour": i["colour"].lower()} for i in items or []
            if isinstance(i, dict) and isinstance(i.get("name"), str) and NAME.fullmatch(i["name"])
            and isinstance(i.get("colour"), str) and COLOUR.fullmatch(i["colour"])]


def remember(root, names=None, forget=None):
    """Add or update names ({name: colour}, newest first) and/or forget names; one entry per name, any case."""
    from .annotation_edits import _atomic
    from .settings import project_lock

    names, forget = names or {}, forget or []
    if not isinstance(names, dict) or not isinstance(forget, list) or len(names) + len(forget) > 200:
        raise ValueError("Give names and colours.")
    for name, colour in names.items():
        if not (isinstance(name, str) and NAME.fullmatch(name.strip())):
            raise ValueError("Names use letters, numbers, spaces and _ . + ( ) - (up to 60).")
        if not (isinstance(colour, str) and COLOUR.fullmatch(colour)):
            raise ValueError("Choose a colour.")
    if not all(isinstance(name, str) for name in forget):
        raise ValueError("Choose names to forget.")
    added = [{"name": n.strip(), "colour": c.lower()} for n, c in names.items()]
    dropped = {n.strip().lower() for n in [*names, *forget]}
    with project_lock(root):
        items = added + [i for i in palette(root) if i["name"].lower() not in dropped]
        _atomic(_palette_path(root), json.dumps({"names": items[:200]}, indent=2).encode())
    return items[:200]


# ---------------------------------------------------------------- several images at once
def _tokens(text):
    """Letters and numbers in order: P11-ROI1 and p11_roi_1 both give p 11 roi 1."""
    return re.findall(r"[a-z]+|\d+", str(text).lower())


def match_image(images, filename):
    """The image a GeoJSON file belongs to. First by name: the image's name, read as words and numbers, appears whole
    in the file's name (P11_roi_1 in "P11-ROI1 old.geojson", never in P11_roi_10's file); the longest wins. Else the
    one image whose name is clearly the closest (fuzzy); else None, to be assigned by hand."""
    import difflib

    from . import annotation

    words = _tokens(Path(str(filename)).name.rsplit(".", 1)[0])
    found = []
    for image in images:
        own = _tokens(annotation.sample_id(Path(image).name))
        if own and any(words[i:i + len(own)] == own for i in range(len(words) - len(own) + 1)):
            found.append((len(own), image))
    if found:
        best = max(item[0] for item in found)
        tied = [image for size, image in found if size == best]
        return tied[0] if len(tied) == 1 else None
    joined = "".join(words)
    scores = sorted(((difflib.SequenceMatcher(None, joined, "".join(_tokens(annotation.sample_id(Path(i).name)))).ratio(), i) for i in images),
                    key=lambda item: -item[0])
    if scores and scores[0][0] >= .85 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= .05):
        return scores[0][1]
    return None


def _annotation_images(root, config):
    from . import annotation

    folder = annotation.all_folders(root, config.get("directory_mappings") or {})["images"]
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith(annotation.IMAGE_EXTENSIONS)) if folder.is_dir() else []


def _stem(name):
    """An imported copy's name as import_polygons writes it, to see a file was imported before."""
    return re.sub(r"[^\w .-]+", "_", Path(str(name)).stem)[:70] or "Imported annotations"


def imports(root, config):
    """Every GeoJSON brought into the project for the Annotation step's images: Studio's imported copies (removable)
    and project files found by an image's name outside the outputs (taken off the image, the file kept)."""
    from . import annotation

    output = annotation.all_folders(root, config.get("directory_mappings") or {})["output"]
    rows = []
    for image in _annotation_images(root, config):
        for entry in files(root, config, str(image), output):
            if entry["origin"] != "imported":
                continue
            rows.append({"image": image.name, "path": entry["path"], "file": entry["source"], "kind": entry["kind"],
                         "names": sorted({name for name, _, _ in _scan(entry)}), "polygons": sum(parts for _, _, parts in _scan(entry))})
    return rows


def remove_imports(root, config, items):
    """Take imported files off their images: Studio's copies leave the project (kept in outputs/studio/deleted); a
    project file found by name is only taken off the image's list. Returns how many were removed."""
    from . import annotation

    if not isinstance(items, list) or not items or len(items) > 5000:
        raise ValueError("Tick the imported files to remove.")
    output = annotation.all_folders(root, config.get("directory_mappings") or {})["output"]
    images = {p.name: p for p in _annotation_images(root, config)}
    removed = 0
    for item in items:
        image = images.get(str((item or {}).get("image", "")))
        if image is None:
            continue
        entry = next((e for e in files(root, config, str(image), output) if e["path"] == str((item or {}).get("path")) and e["origin"] == "imported"), None)
        if entry is None:
            continue
        if entry["kind"] == "imported":
            remove_layer(root, image, output, entry["path"])
        else:
            hide_file(root, image, entry["path"])
        removed += 1
    return removed


def import_directory(root, config, folder, report=lambda *a: None):
    """Every GeoJSON in *folder* imported for the image it names (as several files dropped at once). A file already
    imported for its image (same name) is skipped. An earlier export can be imported (old annotations to reuse)."""
    from . import annotation_inputs as inputs

    folder = Path(str(folder or "").replace("${PROJECT_ROOT}", str(root))).expanduser()
    if not folder.is_absolute():
        folder = Path(root) / folder
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder.")
    found = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".geojson", ".json"))
    if not found:
        raise ValueError(f"{folder} has no GeoJSON files.")
    images = _annotation_images(root, config)
    done, skipped, unmatched = [], [], []
    for number, path in enumerate(found):
        report(f"{path.name} ({number + 1} of {len(found)})", number / len(found))
        image = match_image(images, path.name)
        if image is None:
            unmatched.append(path.name)
            continue
        if any(p.stem.split("_", 1)[-1] == _stem(path.name) for p in inputs.imported(root, image)):
            skipped.append(f"{path.name} (already imported for {image.name})")
            continue
        try:
            data = json.loads(path.read_text())
            features = data if isinstance(data, list) else data.get("features") if data.get("type") == "FeatureCollection" else [data]
            inputs.import_polygons(root, image, path.name, features or [])
            done.append({"file": path.name, "image": image.name})
        except (ValueError, OSError, AttributeError) as error:
            skipped.append(f"{path.name} ({error})")
    return {"imported": done, "skipped": skipped, "unmatched": unmatched, "count": len(done), "folder": str(folder)}


def summary(root, config, images, output):
    """Every annotation across *images*, one row per rule (local or imported name): in which images, how many
    polygons, its colour. Training examples are left out."""
    rows = {}
    for image in images:
        for group in groups(root, config, image, output):
            if not group["rule"]:
                continue
            row = rows.setdefault(group["rule"], {"rule": group["rule"], "name": group["name"], "origin": group["origin"],
                                                  "role": group["role"], "images": set(), "count": 0, "colour": group["colour"]})
            row["images"].add(Path(image).name)
            row["count"] += group["count"]
            row["colour"] = row["colour"] or group["colour"]
    return [{**row, "images": sorted(row["images"])} for row in
            sorted(rows.values(), key=lambda r: (r["origin"] != "local", r["name"].lower()))]


def duplicate(root, image, output, name, origin, features):
    """A copy of a name's polygons as a new layer of the image, under *name* (to rename further)."""
    import uuid

    from . import annotation_edits as edits, annotation_layers
    from .annotation_inputs import image_key

    name = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_ .+()-]{0,59}", name):
        raise ValueError("Names use letters, numbers, spaces and _ . + ( ) - (up to 60).")
    if not isinstance(features, list) or not 0 < len(features) <= 20000:
        raise ValueError("There are no polygons to copy.")
    for feature in features:
        properties = feature.setdefault("properties", {})
        properties.update(name=name, label=name, named=True, copied_from=str(origin or "")[:120])
        cls = properties.get("classification") if isinstance(properties.get("classification"), dict) else {}
        properties["classification"] = {**cls, "name": name}
    target = Path(root) / "outputs/studio/annotation_layers" / image_key(image) / f"copy_{uuid.uuid4().hex[:8]}.geojson"
    edits.replace(target, edits.prepare(features), edits.revision(target), f"Copy of {origin} as {name}")
    annotation_layers.register(root, image, target, name=name, kind="regions", settings={"copied_from": str(origin or "")})
    return annotation_layers.layer(root, image, output, target)


# ---------------------------------------------------------------- removing layers and hiding files
def remove_layer(root, image, output, path):
    """Take a Studio layer (a copy, an imported copy, Marker regions, Classification) out of the project: its file
    moves to outputs/studio/deleted/<image-key>/, so it can be recovered by hand, and it no longer reloads."""
    import time

    from . import annotation_inputs, annotation_layers
    from .annotation_inputs import image_key
    from .settings import project_lock

    path = Path(path).resolve()
    removable = {str(Path(p).resolve()) for p in annotation_layers.generated(root, image, output)}
    removable |= {str(Path(p).resolve()) for p in annotation_inputs.imported(root, image)}
    if str(path) not in removable:
        raise ValueError("This layer is one of the project's own files (tissue, artifacts): delete its polygons instead.")
    target = Path(root) / "outputs/studio/deleted" / image_key(image) / f"{time.strftime('%Y%m%d_%H%M%S')}_{path.name}"
    from . import annotation

    sample = annotation.sample_id(image)
    with project_lock(root):
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        # The whole-tissue classification also wrote the named classes (domains): they go with it.
        domains = annotation.domains(output, sample)
        if path == annotation.classes(output, sample).resolve() and domains.is_file():
            domains.replace(target.parent / f"{time.strftime('%Y%m%d_%H%M%S')}_{domains.name}")
        index = annotation_layers.record_path(root, image)
        records = annotation_layers.records(root, image)
        if str(path) in records:
            records.pop(str(path))
            from .annotation_edits import _atomic

            _atomic(index, json.dumps(records, indent=2).encode())
    return str(target)


# ---------------------------------------------------------------- deleting a stage's results
RESULT_STAGES = ("tissue", "artifacts", "classes", "regions", "zones", "export")


def result_files(root, config, image, stage):
    """The files a stage of the Annotation step saved for one image (what makes its letter green)."""
    from . import annotation, artifact_learning as learning, pieces

    if stage not in RESULT_STAGES:
        raise ValueError("Choose a stage.")
    places = annotation.all_folders(root, config.get("directory_mappings") or {})
    output, sample = places["output"], annotation.sample_id(Path(image).name)
    found = {
        "tissue": lambda: [annotation.valid_tissue(output, sample), annotation.boundary(output, sample),
                           annotation.cleaned_tissue(output, sample), pieces.review(output, sample)],
        "artifacts": lambda: [annotation.detected_artifacts(output, sample), annotation.drawn_artifacts(output, sample),
                              *(learning.polygons_path(output, sample, part) for part in learning.NAMES)],
        "classes": lambda: [annotation.classes(output, sample), annotation.domains(output, sample),
                            *sorted(Path(output).glob(sample + "_classification_*.geojson"))],
        "regions": lambda: [annotation.marker_regions(output, sample), *sorted(Path(output).glob(sample + "_marker_regions_*.geojson"))],
        "zones": lambda: [annotation.zones(places["zones"], sample)],
        "export": lambda: [annotation.exported(places["main"], sample),
                           export_folder(root, export_settings(root, config), config) / file_name(image, export_settings(root, config))],
    }[stage]()
    return [Path(path) for path in dict.fromkeys(found) if Path(path).is_file()]


def delete_results(root, config, image, stage):
    """Set a stage's results for one image aside (outputs/studio/deleted/<image-key>/, to recover by hand): its letter
    turns grey and the stage can run afresh. Classification also forgets its saved run (its settings stay)."""
    import time

    from . import annotation_layers, classification
    from .annotation_edits import _atomic
    from .annotation_inputs import image_key
    from .settings import project_lock

    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(root) / "outputs/studio/deleted" / image_key(image)
    moved = []
    with project_lock(root):
        paths = result_files(root, config, image, stage)
        if stage == "classes":
            runs = classification.directory(root, image)
            if runs.is_dir():
                paths += [p for p in runs.iterdir() if p.is_file() and p.name != "settings.json"]
        if paths:
            folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            target = folder / f"{stamp}_{stage}_{path.name}"
            path.replace(target)
            moved.append(str(target))
        if stage == "classes":
            settings_path = classification.directory(root, image) / "settings.json"
            saved = classification.read_json(settings_path, {})
            if saved.pop("run_id", None) is not None:
                _atomic(settings_path, json.dumps(saved, indent=2).encode())
        records = annotation_layers.records(root, image)
        kept = {path: info for path, info in records.items() if Path(path).is_file()}
        if len(kept) != len(records):
            _atomic(annotation_layers.record_path(root, image), json.dumps(kept, indent=2).encode())
    return moved


def stale_domains(root, image, output):
    """Named classes (domains) left from a Classification deleted before its domains went with it: the classification
    file is gone and was set aside after the domains were written."""
    from . import annotation
    from .annotation_inputs import image_key

    sample = annotation.sample_id(image)
    domains = annotation.domains(output, sample)
    if not domains.is_file() or annotation.classes(output, sample).is_file():
        return False
    folder = Path(root) / "outputs/studio/deleted" / image_key(image)
    name = annotation.classes(output, sample).name
    written = domains.stat().st_mtime
    return any(item.name.endswith("_" + name) and time_of(item) >= written - 5
               for item in (folder.iterdir() if folder.is_dir() else []))


def time_of(item):
    """When a file was set aside: the time in its name (yyyymmdd_HHMMSS_), as seconds."""
    import time

    try:
        return time.mktime(time.strptime(item.name[:15], "%Y%m%d_%H%M%S"))
    except ValueError:
        return 0.0


def _hidden_path(root, image):
    from .annotation_inputs import image_key

    return Path(root) / "outputs/studio/annotation_hidden" / (image_key(image) + ".json")


def hidden_files(root, image):
    """Files taken off this image's annotation list (they stay on disk)."""
    path = _hidden_path(root, image)
    try:
        return set(json.loads(path.read_text())) if path.is_file() else set()
    except ValueError:
        return set()


def hide_file(root, image, path):
    from .annotation_edits import _atomic
    from .settings import project_lock

    with project_lock(root):
        hidden = hidden_files(root, image) | {str(Path(path).resolve())}
        _atomic(_hidden_path(root, image), json.dumps(sorted(hidden), indent=2).encode())
    return hidden
