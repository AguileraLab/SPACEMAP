"""Files brought into the viewer by hand: dropped on the image, or typed as a path (a remote node's folder, say).

A GeoJSON becomes an editable annotation layer (an imported copy; the file is not changed). An image (QPTIFF,
OME-TIFF, TIFF, SVS; PNG, JPG and BMP are turned into a TIFF first) can be opened, or laid over the open image as
its own layer when it has the same size. Dropped files are kept in ``outputs/studio/dropped``."""

import re
import uuid
from pathlib import Path

IMAGES = (".qptiff", ".ome.tiff", ".ome.tif", ".tiff", ".tif", ".svs")
PICTURES = (".png", ".jpg", ".jpeg", ".bmp")
SHAPES = (".geojson", ".json")
TABLES = (".csv", ".tsv", ".parquet")


def kind_of(name):
    name = str(name).lower()
    if name.endswith(SHAPES):
        return "geojson"
    if name.endswith(TABLES):
        return "table"
    if name.endswith(IMAGES) or name.endswith(PICTURES):
        return "image"
    return None


def folder(root):
    return Path(root) / "outputs/studio/dropped"


def target(root, name):
    """Where a dropped file is kept (a new name each time)."""
    if not kind_of(name):
        raise ValueError("Drop a GeoJSON, a cell table (CSV, TSV, Parquet) or an image (QPTIFF, OME-TIFF, TIFF, SVS, PNG or JPG).")
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", Path(str(name)).name)[:120] or "dropped"
    path = folder(root) / f"{uuid.uuid4().hex[:8]}_{safe}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def as_tiff(path):
    """A PNG, JPG or BMP as a TIFF the viewer reads (RGB stays RGB, one channel stays one); others as they are."""
    import numpy as np
    import tifffile
    from PIL import Image

    path = Path(path)
    if not path.name.lower().endswith(PICTURES):
        return path
    with Image.open(path) as picture:
        if picture.mode in ("P", "LA", "RGBA", "CMYK"):
            picture = picture.convert("RGB")
        data = np.asarray(picture)
    out = path.with_name(path.stem + ".ome.tif")
    tifffile.imwrite(out, data, photometric="rgb" if data.ndim == 3 else "minisblack", tile=(256, 256),
                     metadata={"axes": "YXS" if data.ndim == 3 else "YX"})
    return out


def describe(path):
    from spacemap.viewer.sources import image_info

    info = image_info(str(path))
    return {"width": info.width, "height": info.height, "channels": len(info.channel_names), "rgb": bool(info.rgb),
            "name": Path(path).name}


def listing(directory):
    """The GeoJSON files and images in a folder (not inside it), for picking one."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a folder.")
    return sorted(p.name for p in directory.iterdir() if p.is_file() and kind_of(p.name) and not p.name.startswith("."))


def read_shapes(path):
    """A GeoJSON's polygons, as the import takes them."""
    import json

    raw = json.loads(Path(path).read_text())
    features = raw if isinstance(raw, list) else raw.get("features") if raw.get("type") == "FeatureCollection" else [raw] if raw.get("type") == "Feature" else None
    if not features:
        raise ValueError(f"{Path(path).name} has no polygons.")
    return features


def looks_like_cells(features):
    """Numbered cell outlines (a segmentation) rather than annotations: many polygons, nearly all with a number."""
    if len(features) < 50:
        return False
    numbered = sum(1 for f in features[:2000] if isinstance((f.get("properties") or {}).get("label", (f.get("properties") or {}).get("cell_label")), (int, float)))
    return numbered >= 0.9 * min(len(features), 2000)
