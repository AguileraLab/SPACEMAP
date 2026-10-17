"""Studio's Annotation part for tissue pieces (Tab 4 step 2b), live on the image.

The valid tissue is split into its pieces with Tab 4's ``analyse_tissue_pieces``:
narrow bridges (necks) are found and, if asked, cut, and pieces too small for
the analysis are dropped. It takes about a second on a whole ROI, so every
change is shown on the image at once. Applying writes Tab 4's files: the
review layer (kept, dropped, necks) and the valid tissue with only the kept
pieces. Studio always starts from the current Tissue polygons; earlier
versions can be restored explicitly from History.
"""

import json
from pathlib import Path
import time

from . import annotation

KEPT_CLASS, REJECTED_CLASS, NECK_CLASS = "Valid_Tissue", "Rejected_Tissue", "Detected_Neck"
KEPT_COLOUR, REJECTED_COLOUR, NECK_COLOUR = "#2ecc71", "#9e9e9e", "#e91e63"
# The settings Studio shows, with their limits; the rest stay as Tab 4 has them.
EDITABLE = {
    "min_area_mm2": (float, 0.0, 100.0),
    "filter_by_area": (bool, None, None),
    "detect_necks": (bool, None, None),
    "neck_min_width_um": (float, 1.0, 2000.0),
    "neck_min_lobe_area_mm2": (float, 0.0, 10.0),
    "neck_min_lobe_fraction": (float, 0.0, 0.5),
    "cut_at_necks": (bool, None, None),
    "filter_by_inscribed_circle": (bool, None, None),
    "min_inscribed_diameter_um": (float, 1.0, 5000.0),
}


def review(output, sample):
    """Tab 4's review layer for the pieces (kept, dropped, necks)."""
    canonical = Path(output) / annotation.INTERMEDIATE / f"{sample}_tissue_pieces.geojson"
    legacy = Path(output) / f"{sample}_tissue_pieces.geojson"
    return legacy if not canonical.is_file() and legacy.is_file() else canonical


def settings(config):
    """Tab 4's piece settings: its defaults, then the saved ones (as step 2b merges them)."""
    from spacemap.tissue_continuity import DEFAULT_PIECE_CONFIG

    saved = ((config.get("tab_configurations") or {}).get("tab4_annotation") or {}).get("annot_ta_pieces_config")
    return {**DEFAULT_PIECE_CONFIG, **(saved if isinstance(saved, dict) else {})}


def check(changes):
    if not isinstance(changes, dict):
        raise ValueError("Neck settings could not be read. Reload Studio and try again.")
    unknown = set(changes) - set(EDITABLE)
    if unknown:
        raise ValueError("Unrecognised neck setting: " + ", ".join(sorted(unknown)) + ". Reload Studio and try again.")
    for key, value in changes.items():
        kind, low, high = EDITABLE[key]
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Choose on or off for {key}.")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    return changes


def save(root, changes, expected):
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    check(changes)
    with project_lock(root):
        config = read_config(root)
        if expected != revision(settings(config)):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        saved = ((config.get("tab_configurations") or {}).get("tab4_annotation") or {}).get("annot_ta_pieces_config")
        stored = dict(saved) if isinstance(saved, dict) else {}
        stored.update({key: EDITABLE[key][0](value) for key, value in changes.items()})
        persistence.update_stored_config_key(config, "annot_ta_pieces_config", stored)
        persistence.save_project_config(str(root), config)
        return settings(read_config(root))


def polygons(path, classes=None):
    """Every polygon in a GeoJSON, optionally of some classes only (step 2b's ``_polygons_of``)."""
    from shapely.geometry import shape

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    features = data.get("features", []) if isinstance(data, dict) else data
    found = []
    for feature in features:
        if classes is not None:
            classification = (feature.get("properties") or {}).get("classification") or {}
            name = classification.get("name") if isinstance(classification, dict) else classification
            if name not in classes:
                continue
        try:
            geometry = shape(feature["geometry"])
        except Exception:
            continue
        if not geometry.is_empty and geometry.geom_type in ("Polygon", "MultiPolygon"):
            found.append(geometry)
    return found


def source(output, sample, *, path=None, expected=None, class_name=''):
    """Read the current Tissue layer, never resurrect polygons from an old review."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from . import annotation_edits

    path = Path(path) if path is not None else annotation.valid_tissue(output, sample)
    if not path.is_file():
        raise ValueError("Detect tissue first, or load polygons into the Tissue layer.")
    found, version = annotation_edits.read(path, kind="tissue")
    if class_name:
        from .annotation_inputs import annotation_class
        found = [f for f in annotation_edits.read(path)[0] if annotation_class(f) == class_name]
    if expected is not None and version != expected:
        raise ValueError("Tissue changed. Reload saved polygons, then preview necks again.")
    if not found:
        raise ValueError(f"{path.name} has no tissue in it.")
    return unary_union([shape(f["geometry"]) for f in found]), path


def _ring(geometry, limit=2000):
    from spacemap.tissue_continuity import _as_polygons

    rings = []
    for polygon in _as_polygons(geometry):
        for ring in [polygon.exterior, *polygon.interiors]:
            if len(rings) < limit:
                rings.append([[round(x, 1), round(y, 1)] for x, y in ring.coords])
    return rings


def preview(output, sample, values, *, features=None, label="tissue", tissue_path=None, expected=None, class_name=''):
    """Tab 4's pieces with *values*, as outlines to draw; nothing is written."""
    from spacemap.tissue_continuity import analyse_tissue_pieces

    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    from . import annotation_edits

    if features is None:
        geometry, path = source(output, sample, path=tissue_path, expected=expected, class_name=class_name)
        source_name = path.name
    else:
        if not isinstance(features, list) or not features:
            raise ValueError("Detect tissue first, or undo the deletion of all tissue polygons.")
        geometry = unary_union([shape(f["geometry"]) for f in features])
        source_name = "current preview"
    started = time.perf_counter()
    result = analyse_tissue_pieces(geometry, values)
    return {
        "kept": [ring for record in result["kept"] for ring in _ring(record["geom"])],
        "rejected": [ring for record in result["rejected"] for ring in _ring(record["geom"])],
        "necks": [ring for neck in result["necks"] if neck.get("marker") is not None for ring in _ring(neck["marker"])],
        "counts": {"kept": len(result["kept"]), "rejected": len(result["rejected"]), "necks": len(result["necks"])},
        "area_mm2": round(sum(record["area_mm2"] for record in result["kept"]), 3),
        "reasons": [record.get("reason", "") for record in result["rejected"]][:20],
        "source": source_name,
        "microns_per_pixel": values["microns_per_pixel"],
        "features": annotation_edits.prepare([{"type":"Feature", "properties":{}, "geometry":mapping(r["geom"])}
                                               for r in result["kept"]], kind="tissue", label=label) if features is not None else [],
        "neck_regions": [{"id":str(i), "geometry":mapping(n.get("cut_geometry") or n["marker"]), "x":n["x"], "y":n["y"],
                          "width_um":n["bridge_width_um"], "side_areas_um2":[a * 1e6 for a in n["side_areas_mm2"]],
                          "separated":bool(n.get("separated"))}
                         for i, n in enumerate(result["necks"]) if n.get("marker") is not None],
        "ms": round(1000 * (time.perf_counter() - started)),
        "colours": {"kept": KEPT_COLOUR, "rejected": REJECTED_COLOUR, "necks": NECK_COLOUR},
    }


def apply(output, sample, values, *, expected=None, label="tissue", tissue_path=None, kind='tissue', class_name=''):
    """Tab 4 step 2b for one image: the review layer, and the valid tissue with the kept pieces only."""
    from spacemap.tissue_annotation_utils import qupath_color_from_hex, write_feature_collection
    from spacemap.tissue_continuity import analyse_tissue_pieces
    from spacemap_gui.tabs.tab4_annotation.s2b_tissue_pieces import _feature, _neck_feature

    from . import annotation_edits
    tissue = Path(tissue_path) if tissue_path else annotation.valid_tissue(output, sample)
    if expected is None:
        expected = annotation_edits.revision(tissue)

    geometry, path = source(output, sample, expected=expected, path=tissue, class_name=class_name)
    result = analyse_tissue_pieces(geometry, values)
    features = [_feature(record, KEPT_CLASS, KEPT_COLOUR, qupath_color_from_hex) for record in result["kept"]]
    features += [
        _feature(record, REJECTED_CLASS, REJECTED_COLOUR, qupath_color_from_hex) for record in result["rejected"]
    ]
    features += [_neck_feature(neck, index, qupath_color_from_hex) for index, neck in enumerate(result["necks"], 1)]
    target = Path(output) / annotation.INTERMEDIATE / f"{sample}_tissue_pieces.geojson"
    target.parent.mkdir(parents=True, exist_ok=True)
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    from .annotation_inputs import annotation_class
    before, _ = annotation_edits.read(tissue, kind=kind, label=label)
    area = unary_union([record['geom'] for record in result['kept']])
    kept = [{**f,'geometry':mapping(shape(f['geometry']).intersection(area))}
            if not class_name or annotation_class(f)==class_name else f for f in before]
    annotation_edits.replace(tissue, kept, expected,
                             "Applied tissue pieces", kind=kind, label=label)
    write_feature_collection(features, target)
    return {
        "sample": sample,
        "kept": len(result["kept"]),
        "rejected": len(result["rejected"]),
        "necks": len(result["necks"]),
        "source": path.name,
        "files": [str(target), str(tissue)],
    }
