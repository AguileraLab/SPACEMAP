"""Marker regions: a rule's level as raw intensity, the image's Auto or saved level, or an arcsinh z-score of the area."""

import json

import numpy as np
import pytest

from spacemap_gui.studio import regions
from test_viewer_cell_profiles import marker_image


@pytest.fixture
def image(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(20, 3, (2, 120, 120)).clip(0).astype(np.uint16)
    data[0, 30:60, 30:60] = 500  # the positive region, inside the tissue
    data[0, 100:110, 100:110] = 60000  # lint: very bright, outside the tissue
    path = tmp_path / "roi.ome.tiff"
    marker_image(path, data)
    tissue = tmp_path / "tissue.geojson"
    tissue.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [90, 0], [90, 90], [0, 90], [0, 0]]]}}]}))
    return path, tissue


def test_each_cutoff_finds_the_bright_square(image):
    path, tissue = image
    from spacemap.viewer.sources import image_info

    marker = image_info(str(path)).channel_names[0]
    base = {"sigma": 0, "scale": 1, "min_area_um2": 10}  # single noisy pixels are dropped, as in real use
    found = {}
    for cutoff, above, levels in (("raw", 100, None), ("z", 1.5, None), ("auto", 1.0, None), ("saved", 1.0, {marker: 60.0})):
        features, stats = regions.find(path, {**base, "rules": [{"marker": marker, "above": above}], "cutoff": cutoff}, tissue, levels=levels)
        found[cutoff] = stats["cutoffs"][0]
        assert stats["regions"] == 1 and 850 < stats["area_um2"] / 0.25 <= 920, (cutoff, stats)
    assert found["raw"]["raw"] == 100 and found["saved"]["raw"] == 60.0
    # The z-score's middle and spread come from the tissue only: the 60,000 lint outside it does not move the level.
    assert 20 < found["z"]["raw"] < 100 and 15 < found["z"]["median_raw"] < 25
    assert "Triangle" in regions.describe([found["auto"]]) and "z 1.5" in regions.describe([found["z"]])
    with pytest.raises(ValueError, match="no saved level"):
        regions.find(path, {**base, "rules": [{"marker": marker, "above": 1}], "cutoff": "saved"}, tissue, levels={})


def test_cutoff_settings_are_checked():
    assert regions.check({"cutoff": "z", "rules": [{"marker": "CD31", "above": 1.5}]})["cofactor"] == 5.0
    with pytest.raises(ValueError):
        regions.check({"cutoff": "fold", "rules": []})
    with pytest.raises(ValueError):
        regions.check({"cutoff": "auto", "rules": [{"marker": "CD31", "above": 0}]})
    with pytest.raises(ValueError):
        regions.check({"cutoff": "z", "cofactor": 0, "rules": []})


def test_a_ring_keeps_its_dark_middle_as_a_hole(tmp_path):
    """A vessel: bright wall, dark lumen. The region is the wall only, saved and previewed with its hole."""
    from shapely.geometry import shape

    from spacemap.viewer.sources import image_info
    from spacemap_gui.studio import annotation

    data = np.full((2, 120, 120), 20, np.uint16)
    data[0, 20:100, 20:100] = 600
    data[0, 40:80, 40:80] = 20  # the lumen
    path = tmp_path / "vessel.ome.tiff"
    marker_image(path, data)
    marker = image_info(str(path)).channel_names[0]
    features, _ = regions.find(path, {"rules": [{"marker": marker, "above": 100}], "sigma": 0, "scale": 1, "min_area_um2": 0,
                                      "within": "image", "cutoff": "raw"})
    assert len(features) == 1 and len(shape(features[0]["geometry"]).interiors) == 1
    assert len(annotation.rings(features, include_holes=True)) == 2  # the preview draws the hole too


def test_two_groups_split_background_from_signal(image):
    path, tissue = image
    from spacemap.viewer.sources import image_info

    marker = image_info(str(path)).channel_names[0]
    base = {"sigma": 0, "scale": 1, "min_area_um2": 10, "cutoff": "groups", "pool": "image"}
    features, stats = regions.find(path, {**base, "rules": [{"marker": marker, "above": 0.9}]}, tissue)
    found = stats["cutoffs"][0]
    assert stats["regions"] == 1 and 850 < stats["area_um2"] / 0.25 <= 920, stats
    assert 25 < found["raw"] < 500 and found["fitted"] == 1 and found["separation"] > 2, found
    assert "90% sure" in regions.describe([found])
    # Pooled with a second ROI of the slide, the cutoff comes from both, and the same pool gives the same cutoff.
    pool = [(str(path), str(tissue)), (str(path), str(tissue))]
    _, pooled = regions.find(path, {**base, "rules": [{"marker": marker, "above": 0.9}]}, tissue, pool=pool)
    assert pooled["cutoffs"][0]["fitted"] == 2
    with pytest.raises(ValueError):
        regions.check({"cutoff": "groups", "rules": [{"marker": marker, "above": 1.5}]})
    with pytest.raises(ValueError):
        regions.check({"cutoff": "groups", "pool": "patient", "rules": []})


def test_a_slide_pools_its_rois(tmp_path, monkeypatch):
    from spacemap_gui.studio import annotation, annotation_inputs, samples

    images = tmp_path / "images"
    images.mkdir()
    for name in ("S1_roi_1", "S1_roi_2", "S2_roi_1"):
        (images / f"{name}.ome.tiff").write_bytes(b"")
    area = tmp_path / "tissue.geojson"
    area.write_text("{}")
    import pandas as pd

    monkeypatch.setattr(annotation, "all_folders", lambda root, mappings: {"images": images, "output": tmp_path})
    monkeypatch.setattr(annotation_inputs, "scope_path", lambda root, image, output, stage: area)
    monkeypatch.setattr(samples, "_current", lambda root, mappings: (pd.DataFrame(
        {"roi": ["S1_roi_1", "S1_roi_2", "S2_roi_1"], "slide": ["S1", "S1", "S2"]}), {}))
    values = regions.check({"cutoff": "groups", "rules": []})
    pool = regions.pool_for(tmp_path, {}, images / "S1_roi_2.ome.tiff", values)
    assert [p[0].rsplit("/", 1)[-1] for p in pool] == ["S1_roi_1.ome.tiff", "S1_roi_2.ome.tiff"]
    single = regions.pool_for(tmp_path, {}, images / "S1_roi_2.ome.tiff", {**values, "pool": "image"})
    assert len(single) == 1 and regions.pool_for(tmp_path, {}, images / "S1_roi_2.ome.tiff", {**values, "cutoff": "z"}) is None
