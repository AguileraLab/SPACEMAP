"""Image quality learns across markers: a lint taught in some markers is found in markers never trained on, even
faint and among naturally fibrous markers; earlier per-marker examples are measured again when training."""
import numpy as np
import pytest
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import unary_union

from spacemap_gui.studio import artifact_learning as learning
from test_studio_artifacts import marker_image

SIZE, MARKERS = 400, 10
NAMES = ["HE_Red", "HE_Green", "HE_Blue"] + [f"M{i}" for i in range(1, MARKERS + 1)]
TISSUE = [{"type": "Feature", "properties": {}, "geometry": mapping(box(20, 20, 380, 380))}]


def lines(seed, count):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        x0, y0 = r.uniform(40, 360, 2)
        angle, length = r.uniform(0, np.pi), r.uniform(80, 160)
        out.append(LineString([(x0 + t * np.cos(angle) + 8 * np.sin(t / 15), y0 + t * np.sin(angle)) for t in np.linspace(0, length, 30)]))
    return out


def image(path, lint_markers, seed, strength=300.0):
    """Smooth markers, four naturally fibrous ones (M4 to M7), and faint lints in *lint_markers* (0: M1)."""
    r = np.random.default_rng(seed)
    tissue = np.zeros((SIZE, SIZE), np.float32)
    tissue[20:380, 20:380] = 1
    data = [np.full((SIZE, SIZE), 200 - 30 * c, np.float32) + r.normal(0, 5, (SIZE, SIZE)) for c in range(3)]
    for m in range(MARKERS):
        base = ndimage.gaussian_filter(r.gamma(3, 120, (SIZE, SIZE)), 2) * tissue + 50
        if 3 <= m <= 6:
            base += rasterize([(l.buffer(1.0), 1) for l in lines(seed * 10 + m, 40)], out_shape=(SIZE, SIZE)) * 900 * tissue
        data.append(base.astype(np.float32))
    found = lines(seed + 100, 6)
    mask = rasterize([(l.buffer(1.2), 1) for l in found], out_shape=(SIZE, SIZE)).astype(bool)
    for m in lint_markers:
        data[3 + m][mask] += strength
    marker_image(path, np.clip(np.stack(data), 0, 65535).astype(np.uint16), NAMES)
    return unary_union([l.buffer(1.2) for l in found])


def test_a_faint_lint_is_found_in_markers_never_trained_on(tmp_path):
    taught, unseen = tmp_path / "taught.ome.tiff", tmp_path / "unseen.ome.tiff"
    lint = image(taught, [0, 1, 2], 1)
    elsewhere = image(unseen, [7, 8, 9], 2)
    learning.snapshot(tmp_path, "debris", taught, 1, [{"type": "Feature", "properties": {}, "geometry": mapping(lint.buffer(1.5))}], [], TISSUE, True)
    version = learning.train(learning.training_request(tmp_path, "debris"), lambda *a: None)["version"]
    assert version["feature_schema"] == "quality_v2" and not any(n.startswith("marker:") for n in version["feature_names"])
    found = learning.predict(tmp_path, "debris", version["id"], unseen, 1, TISSUE, cut=.5, minimum_um2=0)
    area = unary_union([shape(f["geometry"]) for f in found["features"]])
    assert found["used"] == "Cross-marker quality model"
    assert area.intersection(elsewhere.buffer(2)).area / elsewhere.area > .8  # found where it was never taught
    assert area.difference(elsewhere.buffer(3)).area / (360 * 360) < .03  # and little clean tissue flagged


def test_earlier_examples_are_measured_again_when_training(tmp_path, monkeypatch):
    taught = tmp_path / "taught.ome.tiff"
    lint = image(taught, [0, 1, 2], 1, 900.0)
    original = learning._extractor
    monkeypatch.setattr(learning, "_extractor", lambda image, scale, schema="legacy", **k: original(image, scale, "quality_v1" if schema == "quality_v2" else schema, **k))
    made = learning.snapshot(tmp_path, "debris", taught, 1, [{"type": "Feature", "properties": {}, "geometry": mapping(lint.buffer(1.5))}], [], TISSUE, True)
    monkeypatch.setattr(learning, "_extractor", original)
    record = learning._read(learning.folder(tmp_path, "debris") / (made["snapshot_id"] + ".json"), {})
    record["feature_schema"] = "quality_v1"  # as saved before the cross-marker set
    learning._write(learning.folder(tmp_path, "debris") / (made["snapshot_id"] + ".json"), record)
    request = learning.training_request(tmp_path, "debris")
    request["examples"] = [{**e, "feature_schema": "quality_v1"} for e in request["examples"]]
    result = learning.train(request, lambda *a: None)
    assert result["version"]["feature_schema"] == "quality_v2" and result["left_out"] == []
    assert learning.state(tmp_path, "debris")["examples"][0]["id"] != made["snapshot_id"]  # the library holds the new measurements
