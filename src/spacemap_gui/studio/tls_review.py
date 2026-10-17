"""Manual TLS boundary drafts; saving runs the normal guarded column publication."""
import json

import numpy as np
import shapely
from shapely.geometry import mapping, shape


def polygons(features):
    if not isinstance(features, list) or len(features) > 10000:
        raise ValueError("Send the edited TLS polygons for one tissue region.")
    found = []
    vertices = 0
    for feature in features:
        try:
            geom = shape(feature["geometry"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid TLS boundary.") from exc
        coords = shapely.get_coordinates(geom)
        vertices += len(coords)
        if (geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty or not geom.is_valid
                or not np.isfinite(coords).all() or vertices > 500000):
            raise ValueError("TLS boundaries must be valid polygons with finite coordinates (at most 500,000 vertices).")
        found.append(geom)
    return found


def context(root, table, job_id, image, region):
    from . import jobs, tasks, tls

    job = jobs.status(root, job_id)
    result = job.get("result") or {}
    if job["kind"] != "tls_candidates" or job["status"] != "done" or result.get("table") != str(table):
        raise ValueError("Choose a completed TLS run for this table.")
    if (tls.latest(root, table) or {}).get("job_id") != job_id:
        raise ValueError("TLS results changed. Reopen the latest TLS review before editing.")
    if not any(s["image"] == image and s["region"] == region for s in result["structures"]):
        raise ValueError("Select a TLS in the image and tissue region you want to edit.")
    folder = jobs.directory(root, job_id)
    settings = json.loads((folder / "request.json").read_text())["settings"]
    features = json.loads((folder / "tls_candidates.geojson").read_text())["features"]
    return {"job_id": job_id, "image": image, "region": region,
            "pixel_um": settings["pixel_sizes"][image], "revision": tasks.source_revision(settings),
            "features": [f for f in features if f["properties"]["image"] == image and f["properties"]["region"] == region]}


def prepare(root, table, body):
    from . import jobs, tasks, tls

    current = context(root, table, str(body.get("job_id", "")), str(body.get("image", "")), str(body.get("region", "")))
    if body.get("revision") != current["revision"]:
        raise ValueError("The dataset or tissue annotations changed while editing. Reopen TLS editing before saving.")
    geoms = polygons(body.get("features"))
    previous = jobs.directory(root, current["job_id"])
    settings = json.loads((previous / "request.json").read_text())["settings"]
    settings.pop("review", None)
    settings.update(root=str(root), table=str(table), revision=current["revision"],
                    overwrite_columns=tls.outputs(settings["output_col"]),
                    review={"parent": current["job_id"], "image": current["image"], "region": current["region"],
                            "features": [{"type": "Feature", "geometry": mapping(g), "properties": {}} for g in geoms]})
    tasks.validate_table(settings)
    return settings


def trim(features, amount, pixel_um):
    """An explicit draft operation, independent of B/T connectivity or detection filters."""
    if (isinstance(amount, bool) or not isinstance(amount, (int, float)) or not np.isfinite(amount)
            or not 0 < amount <= 250 or not isinstance(pixel_um, (int, float))
            or not np.isfinite(pixel_um) or pixel_um <= 0):
        raise ValueError("Choose a trim in µm greater than zero and at most 250.")
    from .tls import _parts
    result = []
    for feature, geom in zip(features, polygons(features)):
        for part in _parts(geom.buffer(-amount / pixel_um)):
            result.append({"type": "Feature", "geometry": mapping(part), "properties": feature.get("properties", {})})
    if features and not result:
        raise ValueError("That trim removes every boundary. Choose a smaller trim.")
    return result
