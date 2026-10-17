"""Shared classes: one class model for several images, so a class is the same marker profile in each; names once."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from shapely.geometry import box, mapping

from spacemap_gui.studio import annotation, annotation_edits as edits, class_models, classification, tasks
from test_studio_artifacts import marker_image
from test_studio_pixelclasses import regions  # noqa: F401  (fixture)

NAMES = ["HE_Red", "HE_Green", "HE_Blue", "DAPI", "CD3", "PanCK", "CD68", "CD20"]


@pytest.fixture
def project(regions, tmp_path, monkeypatch):  # noqa: F811
    images, output = tmp_path / "images", tmp_path / "out"
    images.mkdir(); output.mkdir()
    data = tifffile.imread(regions)
    marker_image(images / "A_roi_1.ome.tiff", data, names=NAMES)
    marker_image(images / "B_roi_1.ome.tiff", np.ascontiguousarray(data[:, :, ::-1]), names=NAMES)  # the same tissues, mirrored
    monkeypatch.setattr(annotation, "all_folders", lambda root, mappings: {"images": images, "output": output, "main": output, "zones": output})
    for name in ("A_roi_1.ome.tiff", "B_roi_1.ome.tiff"):
        tissue = annotation.valid_tissue(output, annotation.sample_id(name))
        tissue.parent.mkdir(parents=True, exist_ok=True)
        edits.replace(tissue, [{"type": "Feature", "properties": {}, "geometry": mapping(box(64, 64, 576, 576))}],
                      edits.revision(tissue), "Detected tissue", kind="tissue")
    return tmp_path, images, output


def class_at(root, image, x, y):
    """The class of the pixel at full-resolution (x, y) in the image's last run."""
    last = classification.latest(root, image)
    prepared, found = classification.load(root, image, last["run_id"])
    factor = prepared["factor"]
    return int(found["cluster_map"][y // factor, x // factor])


def test_the_same_class_means_the_same_tissue_in_every_image(project):
    root, images, output = project
    a, b = str(images / "A_roi_1.ome.tiff"), str(images / "B_roi_1.ome.tiff")
    result = tasks.classify_shared({"root": str(root), "images": [a, b], "options": {"scale": 4, "n_clusters": 4}}, None, lambda *x: None)
    model = result["model"]
    assert result["fitted"] and result["count"] == 2 and class_models.listing(root)["selected"] == model
    # CD3 tissue is top-left in A and top-right in B (mirrored): the same class number in both.
    assert class_at(root, a, 150, 150) == class_at(root, b, 640 - 150, 150)
    assert class_at(root, a, 450, 150) == class_at(root, b, 640 - 450, 150)
    assert class_at(root, a, 150, 150) != class_at(root, a, 450, 150)
    assert all(item["far_percent"] < 5 and not item["differ"] for item in result["images"])
    # Names given once (on the model) are every image's names; Save polygons for both writes them.
    cd3 = class_at(root, a, 150, 150)
    state = classification.review({"names": {str(cd3): "T cell zone"}}, list(range(4)))
    class_models.save_review(root, model, state)
    saved = tasks.save_class_polygons({"root": str(root), "images": [a, b], "tidy": {}}, None, lambda *x: None)
    assert saved["count"] == 2
    for image in (a, b):
        features = edits.read(annotation.classes(output, annotation.sample_id(Path(image).name)))[0]
        assert any(f["properties"]["name"] == "T cell zone" for f in features)
    # A new image later: the chosen model classifies it without fitting again.
    again = tasks.classify_shared({"root": str(root), "images": [b], "model": model}, None, lambda *x: None)
    assert not again["fitted"] and again["model"] == model and len(class_models.listing(root)["models"]) == 1


def test_an_image_without_the_models_markers_is_named(project):
    root, images, output = project
    a = str(images / "A_roi_1.ome.tiff")
    record = class_models.fit(root, {}, [a], output, {"scale": 4, "n_clusters": 3})
    data = tifffile.imread(a)
    fewer = images / "C_roi_1.ome.tiff"
    marker_image(fewer, data[:7], names=NAMES[:7])
    tissue = annotation.valid_tissue(output, "C_roi_1")
    edits.replace(tissue, [{"type": "Feature", "properties": {}, "geometry": mapping(box(64, 64, 576, 576))}], edits.revision(tissue), "t", kind="tissue")
    with pytest.raises(ValueError, match="lacks CD20"):
        class_models.apply(root, {}, str(fewer), output, record["id"])
    with pytest.raises(ValueError, match="saved class model"):
        class_models.read(root, "../x")
    json.dumps(record)


def test_per_slide_scaling_keeps_a_one_tissue_image_comparable(project):
    """A_roi_2 (same slide as A_roi_1) has only the CD3 area as tissue. Scaled per slide, its pixels keep their
    level among the slide's, so they get A_roi_1's CD3 class."""
    root, images, output = project
    import shutil

    shutil.copyfile(images / "A_roi_1.ome.tiff", images / "A_roi_2.ome.tiff")
    tissue = annotation.valid_tissue(output, "A_roi_2")
    edits.replace(tissue, [{"type": "Feature", "properties": {}, "geometry": mapping(box(64, 64, 320, 320))}], edits.revision(tissue), "t", kind="tissue")
    a, a2, b = (str(images / n) for n in ("A_roi_1.ome.tiff", "A_roi_2.ome.tiff", "B_roi_1.ome.tiff"))
    assert class_models.slide_of(a2) == "A" and class_models.slide_of(b) == "B"
    result = tasks.classify_shared({"root": str(root), "images": [a, a2, b], "options": {"scale": 4, "n_clusters": 4}, "scale_by": "slide"}, None, lambda *x: None)
    record, _ = class_models.read(root, result["model"])
    assert record["scale_by"] == "slide" and set(record["slide_scales"]) == {"A", "B"}
    assert class_at(root, a2, 150, 150) == class_at(root, a, 150, 150) == class_at(root, b, 640 - 150, 150)
    with pytest.raises(ValueError, match="marker intensity only"):
        class_models.fit(root, {}, [a], output, {"scale": 4, "n_clusters": 3, "features": ["texture"]}, scale_by="slide")
