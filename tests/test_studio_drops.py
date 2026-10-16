"""Files brought in by hand: a typed folder lists its files; a PNG becomes a TIFF the viewer opens; a GeoJSON gives its polygons."""

import json

import numpy as np
import pytest
from PIL import Image

from spacemap_gui.studio import drops


def test_a_folder_lists_geojson_and_images_and_a_png_becomes_a_tiff(tmp_path):
    (tmp_path / "cells.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 0]]]}}]}))
    Image.fromarray(np.arange(64 * 48, dtype=np.uint8).reshape(48, 64)).save(tmp_path / "mask.png")
    (tmp_path / "notes.txt").write_text("x")
    assert drops.listing(tmp_path) == ["cells.geojson", "mask.png"]
    tiff = drops.as_tiff(tmp_path / "mask.png")
    facts = drops.describe(tiff)
    assert tiff.name == "mask.ome.tif" and (facts["width"], facts["height"]) == (64, 48)
    assert len(drops.read_shapes(tmp_path / "cells.geojson")) == 1
    with pytest.raises(ValueError):
        drops.target(tmp_path, "virus.exe")
    with pytest.raises(ValueError):
        drops.listing(tmp_path / "missing")
