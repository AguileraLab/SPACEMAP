"""Artifacts for the ticked images: found with the chosen model in each image's Work within, saved (earlier ones in
History), kept unless Replace is on, and subtracted from the tissue when asked."""
import shutil
from pathlib import Path

import pytest

from spacemap_gui.studio import annotation, annotation_edits as edits, artifact_learning as learning, tasks
from test_artifact_learning import polygon, slide  # noqa: F401  (fixture)


@pytest.fixture
def project(tmp_path, slide, monkeypatch):
    image, _ = slide
    images, output = tmp_path / "images", tmp_path / "out"
    images.mkdir(); output.mkdir()
    for name in ("a.ome.tiff", "b.ome.tiff"):
        shutil.copyfile(image, images / name)
    monkeypatch.setattr(annotation, "all_folders", lambda root, mappings: {"images": images, "output": output, "main": output, "zones": output})
    for name in ("a.ome.tiff", "b.ome.tiff"):
        tissue = annotation.valid_tissue(output, annotation.sample_id(name))
        edits.replace(tissue, [polygon()], edits.revision(tissue), "Detected tissue", kind="tissue")
    learning.snapshot(tmp_path, "clot", images / "a.ome.tiff", 2, [polygon(0, 0, 64, 128, "Blood_Clot")], [polygon(70, 0, 128, 128)], [polygon()], False)
    version = learning.train(learning.training_request(tmp_path, "clot"), lambda *a: None)["version"]["id"]
    learning.save_options(tmp_path, "clot", {"scale": 2, "minimum_um2": 0}, learning.state(tmp_path, "clot")["revision"])
    return tmp_path, images, output, version


def run(root, images, **settings):
    return tasks.artifact_batch({"root": str(root), "part": "clot", "images": [str(images / n) for n in ("a.ome.tiff", "b.ome.tiff")], **settings},
                                None, lambda *a: None)


def test_finds_saves_keeps_and_subtracts(project):
    root, images, output, version = project
    found = run(root, images, version=version, find=True)
    assert [line["image"] for line in found["images"]] == ["a.ome.tiff", "b.ome.tiff"] and all(line["found"] for line in found["images"])
    saved = learning.polygons_path(output, "a", "clot")
    polygons, revision = edits.read(saved, kind="artifacts")
    assert polygons and learning._geometry(polygons).intersection(learning._geometry([polygon(0, 0, 64, 128)])).area > 0
    # Saved polygons are kept unless Replace is on; with it they are found again and the earlier ones stay in History.
    again = run(root, images, version=version, find=True)
    assert again["kept"] == ["a.ome.tiff", "b.ome.tiff"] and edits.revision(saved) == revision
    run(root, images, version=version, find=True, replace=True)
    assert len(edits.history(saved)["entries"]) >= 2
    tissue = annotation.valid_tissue(output, "a")
    before = learning._geometry(edits.read(tissue, kind="tissue")[0]).area
    done = run(root, images, subtract=True)
    assert all(line["subtracted"] for line in done["images"])
    assert learning._geometry(edits.read(tissue, kind="tissue")[0]).area < before


def test_blood_clots_need_a_model_and_a_failed_image_does_not_stop_the_rest(project):
    root, images, output, version = project
    with pytest.raises(ValueError, match="model"):
        run(root, images, find=True)
    annotation.valid_tissue(output, "b").unlink()
    found = run(root, images, version=version, find=True)
    assert [line["image"] for line in found["images"]] == ["a.ome.tiff"] and found["failed"][0]["image"] == "b.ome.tiff"


def test_train_on_the_ticked_reviewed_images(project):
    root, images, output, version = project
    run(root, images, version=version, find=True)
    # The user fixes b's prediction by hand: one polygon only, the clot half of the image.
    fixed = learning.polygons_path(output, "b", "clot")
    edits.replace(fixed, [polygon(0, 0, 60, 128, "Blood_Clot")], edits.revision(fixed), "Edited by hand", kind="artifacts")
    never = images / "c.ome.tiff"
    shutil.copyfile(images / "a.ome.tiff", never)  # ticked but never reviewed: left out
    settings = learning.training_request(root, "clot", base_version=version, allow_empty=True)
    captures, skipped = learning.batch_captures(root, "clot", [images / "a.ome.tiff", images / "b.ome.tiff", never], output, 2, True)
    assert [Path(c["image"]).name for c in captures] == ["a.ome.tiff", "b.ome.tiff"] and "c.ome.tiff" in skipped[0]
    assert learning._geometry(captures[1]["positives"]).area == 60 * 128
    result = tasks.train_artifact_model({**settings, "captures": captures, "skipped": skipped}, None, lambda *a: None)
    trained = result["version"]
    assert trained["images"] == 2 and trained["parent_version"] == version and len(result["trained_on"]) == 2
    assert {e["image"] for e in learning.state(root, "clot")["examples"]} == {"a.ome.tiff", "b.ome.tiff"}
    assert learning.predict(root, "clot", trained["id"], images / "b.ome.tiff", 2, [polygon()], minimum_um2=0)["features"]


def test_both_kinds_are_subtracted_together(project):
    root, images, output, version = project
    for part, square in (("clot", (0, 0, 20, 20)), ("debris", (100, 100, 128, 128))):
        saved = learning.polygons_path(output, "a", part)
        edits.replace(saved, [polygon(*square, learning.CLASSES[part])], edits.revision(saved), "Edited by hand", kind="artifacts")
    tissue = annotation.valid_tissue(output, "a")
    done = tasks.artifact_batch({"root": str(root), "part": "clot", "parts": ["clot", "debris"], "subtract": True,
                                 "images": [str(images / "a.ome.tiff")]}, None, lambda *a: None)
    left = learning._geometry(edits.read(tissue, kind="tissue")[0])
    assert done["images"][0]["subtracted"] == 2 and left.area == 128 * 128 - 400 - 28 * 28


def test_small_tissue_pieces_are_removed_in_each_image(project):
    root, images, output, version = project
    for name in ("a", "b"):
        tissue = annotation.valid_tissue(output, name)
        edits.replace(tissue, [polygon(), polygon(200, 200, 204, 204)], edits.revision(tissue), "two pieces", kind="tissue")
    from spacemap.viewer.sources import image_info
    microns = image_info(str(images / "a.ome.tiff")).pixel_size_um or 0.5083
    done = tasks.remove_small_pieces({"root": str(root), "min_um2": 100 * microns ** 2,
                                      "images": [str(images / n) for n in ("a.ome.tiff", "b.ome.tiff")]}, None, lambda *a: None)
    assert [item["removed"] for item in done["images"]] == [1, 1]
    for name in ("a", "b"):
        tissue = annotation.valid_tissue(output, name)
        assert len(edits.read(tissue, kind="tissue")[0]) == 1 and len(edits.history(tissue)["entries"]) >= 2
