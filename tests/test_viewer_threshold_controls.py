"""Scientific and integration checks for marker-based threshold controls."""

import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from shapely.geometry import box

from spacemap.background_detection import THRESHOLD_METHODS, find_background
from spacemap.viewer import display, live_resolve, render, transforms
from spacemap.viewer.sources import open_source
from spacemap_gui.viewer import registry
from test_viewer import _square, _write_channel_pages, client  # noqa: F401


@pytest.mark.parametrize(
    "steps",
    [
        [],
        ["Log10"],
        ["Arcsinh"],
        ["Z1"],
        ["Arcsinh", "Z1"],
        ["Log10", "Z1", "CDF"],
        ["Arcsinh", "Z1", "CDF", "NLog"],
        ["Z1", "Arcsinh", "Log10"],
    ],
)
def test_histogram_transform_chains_preserve_raw_thresholds(steps):
    values = np.r_[0.0, np.linspace(0.025, 65535.0, 2000)]
    transform = transforms.Transform.fit(values, steps)
    thresholds = np.r_[0.0, 0.1, 127.375, 65535.0, np.linspace(1, 65534, 100)]
    np.testing.assert_allclose(transform.inverse(transform.forward(thresholds)), thresholds, atol=1e-6, rtol=1e-10)


def test_constant_and_negative_features_have_reversible_views():
    for values in (np.full(30, 10.5), np.linspace(-200, 200, 150)):
        chain = transforms.Transform.fit(values, ["Log10", "Arcsinh", "Z1"])
        np.testing.assert_allclose(chain.inverse(chain.forward(values)), values, atol=1e-6)


@pytest.mark.parametrize("steps", [["Z2"], ["CDF"], ["NLog"], ["Raw"] * 9])
def test_lossy_or_undefined_chains_are_rejected(steps):
    with pytest.raises(ValueError):
        transforms.Transform.fit([0, 100, 65535], steps)


def test_cdf_does_not_silently_clip_out_of_domain_thresholds():
    chain = transforms.Transform.fit(np.linspace(-1, 1, 100), ["CDF"])
    with pytest.raises(ValueError):
        chain.inverse(1)


@pytest.mark.parametrize("method", THRESHOLD_METHODS)
def test_auto_uses_the_same_pixels_and_detector_as_background_detection(tmp_path, method):
    image = tmp_path / "image.ome.tiff"
    _write_channel_pages(image, channels=1, height=120, width=160)
    info = open_source(image).info
    region = box(12, 10, 75, 80)
    pixels, _ = render.marker_pixels(str(image), info, 0, region=region)
    expected = find_background(pixels, method, imagej_dtype=pixels.dtype)
    actual = render.background_level(str(image), info, 0, method, region=region)
    assert actual["threshold"] == expected["threshold"]
    assert actual["pixels"] == pixels.size


def test_auto_region_routes_validate_image_class_and_empty_selection(client, tmp_path):
    image = tmp_path / "image.ome.tiff"
    _write_channel_pages(image, channels=1, height=120, width=160)
    token = registry.register(image)
    path = tmp_path / "tissue.geojson"
    feature = _square(1, 10, 10, 30)
    feature["properties"] = {"classification": {"name": "tissue"}}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    annotation = registry.register_object("annotations", path=str(path), image=str(image))
    route = f"/image/{token}/background"
    query = {"ch": 0, "method": "Otsu", "annotation": annotation, "class": "tissue"}
    result = client.get(route, params=query)
    assert result.status_code == 200
    assert result.json()["region"] is True and result.json()["pixels"] < 120 * 160
    query["class"] = "missing"
    assert client.get(route, params=query).status_code == 400
    wrong = registry.register_object("annotations", path=str(path), image=str(tmp_path / "other.tiff"))
    assert client.get(route, params={"ch": 0, "annotation": wrong}).status_code == 400
    assert client.get(route, params={"ch": 0, "method": "not-a-method"}).status_code == 400
    assert client.get(route, params={"ch": 5}).status_code == 400
    assert client.get("/image/unregistered/background?ch=0").status_code == 404


def test_auto_settings_persist_without_breaking_older_writers(tmp_path):
    path = tmp_path / "display.json"
    channels = [{"index": 0, "name": "CD8", "low": 77.125, "high": 500}]
    options = {"method": "Li", "scope": json.dumps({"annotation": "signed-token", "class": "tissue"})}
    display.save(path, channels, options)
    assert display.load(path)["auto"] == options
    display.save(path, channels)
    assert display.load(path)["auto"] == options
    assert display.minimums(path) == {"CD8": 77.125}


def test_threshold_updates_are_atomic_and_finite():
    session = SimpleNamespace(lock=threading.Lock(), thresholds={"CD8": 50.0}, source={"CD8": "saved"})
    for changes in ({"CD8": 20, "unknown": 7}, {"CD8": float("nan")}, {"CD8": float("inf")}):
        with pytest.raises(ValueError):
            live_resolve.assign(session, changes)
        assert session.thresholds == {"CD8": 50.0}
        assert session.source == {"CD8": "saved"}


def test_histogram_cache_reuses_bins_but_refreshes_threshold_and_count(monkeypatch):
    session = SimpleNamespace(
        lock=threading.Lock(),
        markers=["CD8"],
        raw=np.arange(1000).reshape(-1, 1),
        thresholds={"CD8": 10.0},
        histogram_transforms={},
        histogram_views={},
    )
    initial = live_resolve.transformed_histogram(session, "CD8", ["Arcsinh", "Z1"])

    def unexpected(*args, **kwargs):
        raise AssertionError("Threshold dragging must reuse histogram bins")

    monkeypatch.setattr(transforms, "histogram", unexpected)
    session.thresholds["CD8"] = 200.5
    changed = live_resolve.transformed_histogram(session, "CD8", ["Arcsinh", "Z1"])
    assert initial["counts"] == changed["counts"]
    assert changed["positive"] == 799
    assert changed["threshold"] == 200.5


@pytest.mark.parametrize("steps", [[], ["Arcsinh"], ["Arcsinh", "Z1", "CDF", "NLog"]])
def test_auto_on_transformed_histogram_converts_back_to_raw(steps):
    rng = np.random.default_rng(0)
    values = rng.lognormal(5, 1, 10000)
    transform = transforms.Transform.fit(values, steps)
    result = transforms.automatic_threshold(values, transform, "Otsu")
    assert result["raw"] == pytest.approx(float(transform.inverse(result["value"])))
    assert values.min() <= result["raw"] <= values.max()
