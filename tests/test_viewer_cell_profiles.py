"""Measured cell pixels must exclude neighbours and include every outline part."""

from io import StringIO
import json
from pathlib import Path
import time

import numpy as np
import pytest
import tifffile

from spacemap.viewer import cell_profiles as profiles
from spacemap.viewer import cells
from spacemap.viewer.sources import open_source


def polygon(label, ring, holes=()):
    return {
        "type": "Feature",
        "properties": {"label": label},
        "geometry": {"type": "Polygon", "coordinates": [ring, *holes]},
    }


def square(x, y, size):
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]


def marker_image(path, data, names=None):
    count, height, width = data.shape
    xml = (
        f'<OME><Image><Pixels DimensionOrder="XYCZT" Type="uint16" SizeX="{width}" '
        f'SizeY="{height}" SizeC="{count}" SizeZ="1" SizeT="1">'
        + "".join(f'<Channel Name="{name}"/>' for name in (names or [f"M{i}" for i in range(count)]))
        + "</Pixels></Image></OME>"
    )
    with tifffile.TiffWriter(path) as writer:
        for i, channel in enumerate(data):
            writer.write(channel, subifds=1, tile=(16, 16), metadata=None, description=xml if i == 0 else None)
            writer.write(channel[::2, ::2], subfiletype=1, tile=(16, 16), metadata=None)


@pytest.fixture
def image_cells(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    source = tmp_path / "cells.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    polygon(1, square(2, 2, 10), [square(5, 5, 2)]),
                    polygon(1, square(20, 20, 3)),  # separate piece of the same numbered cell
                    polygon(2, square(12, 2, 4)),
                ],
            }
        )
    )
    pyramid = cells.ensure_label_pyramid(source, 32, 32, tmp_path / "outlines")
    labels = open_source(pyramid).read_region(0, 0, 0, 0, 32, 32)
    data = np.full((2, 32, 32), 60000, dtype=np.uint16)
    data[0][labels == 1] = np.arange(np.sum(labels == 1)) % 20
    data[1][labels == 1] = 7
    image = tmp_path / "roi.ome.tiff"
    marker_image(image, data)
    profiles.build_index(source, profiles.index_path(source))
    return image, pyramid, source, labels, data


@pytest.mark.parametrize("chunk_size", [1, 3, 11, 65536])
def test_streaming_parser_handles_split_values_and_unicode(chunk_size):
    features = [polygon(12, square(123.456, 2, 3)), polygon(3, square(-0.12345, 4, 5))]
    features[0]["properties"]["note"] = 'tissué "boundary"'
    for value in (features, {"type": "FeatureCollection", "features": features}, features[0]):
        result = list(profiles.iter_features(StringIO(json.dumps(value, ensure_ascii=False)), chunk_size))
        assert result == (features if value != features[0] else [features[0]])
    # Top-level metadata may split just after a decimal point or exponent.
    document = '{"version": 123.45e-6, "features": ' + json.dumps(features) + "}"
    assert list(profiles.iter_features(StringIO(document), chunk_size)) == features


def test_index_unites_all_parts_of_a_cell(image_cells):
    _, _, source, _, _ = image_cells
    with profiles._connect(profiles.index_path(source)) as db:
        assert db.execute("SELECT * FROM bounds WHERE label=1").fetchone() == (1, 2, 2, 23, 23)


def test_pixel_profile_is_exact_and_excludes_holes_and_neighbours(image_cells):
    image, pyramid, source, labels, data = image_cells
    result = profiles.image_profile(image, pyramid, source, 1, {"M0": 10.5, "M1": 2})
    assert result["status"] == "ready"
    assert result["pixels"] == np.sum(labels == 1)
    for channel, row in enumerate(result["markers"]):
        expected = data[channel][labels == 1].astype(float)
        background = [10.5, 2][channel]
        assert row["mean"] == pytest.approx(expected.mean())
        assert row["signal"] == pytest.approx(np.maximum(expected - background, 0).mean())
        assert row["positive_percent"] == pytest.approx(100 * np.mean(expected > background))
        assert row["mean"] < 60000  # neighbouring and outside pixels never contribute


def test_unset_background_is_reported_instead_of_assuming_zero(image_cells):
    image, pyramid, source, _, _ = image_cells
    result = profiles.image_profile(image, pyramid, source, 1, {"M0": 10})
    assert result["missing_backgrounds"] == ["M1"]
    assert result["markers"][1]["signal"] is None
    assert result["markers"][1]["positive_percent"] is None


def test_profiles_include_nuclear_markers_and_skip_he_channels(image_cells, tmp_path):
    import pandas as pd
    from spacemap_gui.viewer.api import _markers

    image, pyramid, source, _, _ = image_cells
    marker_image(image, np.full((3, 32, 32), 7, np.uint16), ["DAPI", "CD3", "HE_R"])
    table = tmp_path / "table.csv"
    pd.DataFrame({"DAPI": [7], "CD3": [7], "HE_R": [7]}).to_csv(table, index=False)
    assert _markers({"image": str(image)}, str(table), include_nuclear=True) == ["DAPI", "CD3"]
    result = profiles.image_profile(image, pyramid, source, 1, {}, corrected=True)
    assert [m["marker"] for m in result["markers"]] == ["DAPI", "CD3"]


def test_full_resolution_measurement_does_not_open_unused_pyramid_levels(image_cells, monkeypatch):
    from spacemap.viewer.sources import ImageSource

    image, _, _, _, data = image_cells
    source = ImageSource(image)

    def no_levels(*args):
        raise AssertionError("Unneeded image level was opened")

    monkeypatch.setattr(source, "_levels_of", no_levels)
    try:
        np.testing.assert_array_equal(source.read_region(0, 1, 0, 0, 16, 16), data[1, :16, :16])
    finally:
        source.close()


def test_corrected_image_uses_zero_without_subtracting_twice(image_cells):
    image, pyramid, source, labels, data = image_cells
    result = profiles.image_profile(image, pyramid, source, 1, {"M0": 5000}, corrected=True)
    assert result["source"] == "corrected"
    assert not result["missing_backgrounds"]
    assert result["markers"][0]["signal"] == pytest.approx(data[0][labels == 1].mean())
    assert result["markers"][1]["positive_percent"] == 100


def test_changed_background_changes_cached_measurement(image_cells):
    image, pyramid, source, _, _ = image_cells
    first = profiles.image_profile(image, pyramid, source, 1, {"M0": 5})
    assert profiles.image_profile(image, pyramid, source, 1, {"M0": 5}) is first
    changed = profiles.image_profile(image, pyramid, source, 1, {"M0": 10})
    assert changed["markers"][0]["signal"] < first["markers"][0]["signal"]


def test_changed_outlines_reject_old_pixel_masks(image_cells):
    image, pyramid, source, _, _ = image_cells
    source.write_text(source.read_text() + " ")
    assert not profiles.index_current(source, profiles.index_path(source))
    with pytest.raises(ValueError, match="out of date"):
        profiles.image_profile(image, pyramid, source, 1, {})


def test_mismatched_image_dimensions_are_rejected(image_cells, tmp_path):
    _, pyramid, source, _, _ = image_cells
    image = tmp_path / "other.ome.tiff"
    marker_image(image, np.ones((2, 64, 64), np.uint16))
    with pytest.raises(ValueError, match="does not line up"):
        profiles.image_profile(image, pyramid, source, 1, {})


def test_bad_index_input_does_not_replace_good_index(image_cells):
    _, _, source, _, _ = image_cells
    target = profiles.index_path(source)
    previous = target.read_bytes()
    source.write_text('{"features": [broken')
    with pytest.raises(ValueError):
        profiles.build_index(source, target)
    assert target.read_bytes() == previous


def test_background_image_discovery_does_not_treat_raw_rois_as_corrected(tmp_path):
    from spacemap.viewer import catalog

    rois = tmp_path / "outputs/rois"
    rois.mkdir(parents=True)
    raw = rois / "roi.ome.tiff"
    raw.touch()
    entry = catalog.ImageEntry("ROIs", raw.name, str(raw), "roi")
    mapping = {"tab2_output_dir": str(rois)}
    assert catalog.background_image_for(tmp_path, mapping, entry) is None
    corrected = rois / "roi_bg_subtracted.ome.tiff"
    corrected.touch()
    assert Path(catalog.background_image_for(tmp_path, mapping, entry)) == corrected


def test_index_preparation_runs_in_a_child_and_survives_app_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    source = tmp_path / "cells.geojson"
    source.write_text(json.dumps({"features": [polygon(1, square(1, 1, 4))]}))
    path, progress = profiles.prepare_index(source)
    assert progress["status"] == "preparing"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        path, progress = profiles.prepare_index(source)
        if progress["status"] != "preparing":
            break
        time.sleep(0.05)
    assert progress["status"] == "ready", progress
    assert profiles.index_current(source, path)


@pytest.mark.parametrize("fast_copy", [False, True])
def test_profile_api_selects_sources_and_reads_saved_table_values(image_cells, tmp_path, monkeypatch, fast_copy):
    import pandas as pd
    from starlette.testclient import TestClient

    from spacemap import dataio, labels
    from spacemap_gui.viewer import api, registry

    image, pyramid, source, _, _ = image_cells
    table = tmp_path / "cells.csv"
    pd.DataFrame(
        {"batch_id": ["roi", "different_roi"], "label": [1, 1], "M0": [12.345678901, 9000], "M1": [33, 9000]}
    ).to_csv(table, index=False)
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on" if fast_copy else "off")
    if fast_copy:
        assert dataio.ensure_fast_copy(table, force=True)
    labels.append(table, [{"row": 0, "column": "phenotype_ref1", "value": "T cell"}])
    token = registry.register_object(
        "cells",
        image=str(image),
        pyramid=str(pyramid),
        segmentation=str(source),
        background_image=str(image),
        roi="roi",
        tables={"t0": str(table)},
    )
    with TestClient(api.app) as client:
        route = f"/cells/{token}/profile"
        saved = client.post(route, json={"label": 1, "source": "table", "t": "t0"})
        assert saved.status_code == 200
        assert saved.json()["markers"][0]["mean"] == pytest.approx(12.345678901, abs=1e-10)
        pixels = client.post(route, json={"label": 1, "source": "image", "backgrounds": {"M0": 10, "M1": 1}})
        assert pixels.status_code == 200
        assert pixels.json()["source"] == "image"
        corrected = client.post(route, json={"label": 1, "source": "corrected"})
        assert corrected.status_code == 200
        assert corrected.json()["markers"][0]["background"] == 0
        assert client.post(route, json={"label": 1, "source": "invented"}).status_code == 400
        assert client.post(route, json=["bad body"]).status_code == 400
        for bad_label in (0, -1, 1.5, True, 2**32, "not a cell"):
            assert client.post(route, json={"label": bad_label}).status_code == 400
        assert client.post("/cells/unregistered/profile", json={"label": 1}).status_code == 404
