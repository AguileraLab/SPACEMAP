"""Studio's Cellpose training keeps corrected regions as Cellpose examples and trains a new model on them."""

import json

import numpy as np
import pytest

from spacemap_gui.studio import training
from test_studio_artifacts import marker_image


def disc(cx, cy, radius, points=24):
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
    return [[float(cx + radius * np.cos(a)), float(cy + radius * np.sin(a))] for a in angles]


@pytest.fixture
def project(tmp_path):
    """An image of round nuclei (DAPI) with a faint border marker, and the nuclei's outlines."""
    rng = np.random.default_rng(2)
    size = 256
    yy, xx = np.mgrid[:size, :size]
    dapi = rng.normal(200, 30, (size, size))
    border = rng.normal(150, 20, (size, size))
    centres = [(x, y) for x in range(24, size - 20, 36) for y in range(24, size - 20, 36)]
    for x, y in centres:
        inside = (xx - x) ** 2 + (yy - y) ** 2 < 11**2
        dapi[inside] += 3000
        ring = np.abs(np.sqrt((xx - x) ** 2 + (yy - y) ** 2) - 13) < 2
        border[ring] += 800
    root = tmp_path / "study"
    (root / "outputs").mkdir(parents=True)
    image = root / "S1_roi_1.ome.tiff"
    marker_image(image, np.clip(np.stack([dapi, border]), 0, 65535).astype(np.uint16), names=["DAPI", "CD45"])
    return root, str(image), [disc(x, y, 11) for x, y in centres]


SETTINGS = {"tab4_last_nuclear_names": ["DAPI"], "tab4_last_membrane_names": []}


def test_the_corrected_cells_become_a_mask():
    masks = training.masks_from([disc(20, 20, 8), disc(26, 20, 8), disc(500, 500, 5), [[1, 1], [2, 2]]], [0, 0, 64, 48])
    assert masks.shape == (48, 64) and masks.dtype == np.uint16
    assert set(np.unique(masks)) == {0, 1, 2}  # the cell outside and the line are dropped
    assert masks[20, 27] == 2 and masks[20, 13] == 1  # the later cell wins the overlap
    for wrong in ([0, 0, 10, 10], [0, 0, 2000, 50], [-5, 0, 64, 64], "box"):
        with pytest.raises(ValueError):
            training.check_region(wrong, 1000, 1000)


def test_tiles_are_kept_with_their_prediction_and_role(project):
    import tifffile

    root, image, cells = project
    inside = [ring for ring in cells if all(20 <= x <= 140 and 20 <= y <= 140 for x, y in ring)]
    predicted = inside[:-1]  # the model missed one cell
    found = training.save_example(
        root,
        image,
        ["DAPI", "CD45"],
        [20, 20, 120, 120],
        inside,
        SETTINGS,
        role="validate",
        predicted=predicted,
        model="nuclei",
    )
    folder, _ = training.folders(root)
    name = found["saved"]
    picture, masks = tifffile.imread(folder / f"{name}_img.tif"), tifffile.imread(folder / f"{name}_masks.tif")
    assert picture.shape == (3, 120, 120) and masks.shape == (120, 120)
    assert found["saved_cells"] == len(inside) == int(masks.max()) and found["counts"] == {"train": 0, "validate": 1}
    assert picture[2].max() > 2000 and not picture[1].any()  # nuclear in the third plane, the second empty
    assert int(tifffile.imread(folder / f"{name}_pred.tif").max()) == len(predicted)
    checked = training.validate(root)
    tile = checked["tiles"][0]
    assert (tile["tp"], tile["fp"], tile["fn"]) == (len(predicted), 0, 1) and tile["precision"] == 1.0
    assert tile["recall"] == pytest.approx(len(predicted) / len(inside), abs=1e-3) and checked["overall"]["f1"] < 1
    assert training.set_role(root, [name], "train")["counts"] == {"train": 1, "validate": 0}
    with pytest.raises(ValueError):
        training.validate(root)
    assert training.delete_example(root, name)["examples"] == []
    with pytest.raises(ValueError):
        training.save_example(root, image, ["DAPI", "CD45"], [20, 20, 120, 120], [], SETTINGS)
    with pytest.raises(ValueError):
        training.save_example(root, image, ["DAPI", "CD45"], [20, 20, 120, 120], inside, SETTINGS, role="test")


def test_each_round_trains_the_next_version_from_the_current_one(project):
    root, image, cells = project
    with pytest.raises(ValueError, match="session"):
        training.train({"project_root": str(root)}, root / "job", lambda *args: None)
    started = training.start(root, "studio_test", "nuclei")
    assert started["current"] == "nuclei" and started["versions"] == []
    for x, y, role in ((0, 0, "train"), (128, 0, "train"), (0, 128, "train"), (128, 128, "validate")):
        inside = [ring for ring in cells if all(x <= px < x + 128 and y <= py < y + 128 for px, py in ring)]
        training.save_example(
            root, image, ["DAPI", "CD45"], [x, y, 128, 128], inside, SETTINGS, role=role, predicted=inside
        )
    settings = {"project_root": str(root), "n_epochs": 1, "gpu": False}
    first = training.train(settings, root / "job", lambda *args: None)
    assert first["name"] == "studio_test_v1" and first["from"] == "nuclei" and first["training_tiles"] == 3
    assert first["validation_tiles"] == 1 and 0 <= first["ap_after"] <= 1
    second = training.train(settings, root / "job2", lambda *args: None)
    assert second["name"] == "studio_test_v2" and second["from"] == first["model"]
    state = training.session(root)
    assert state["current"] == second["model"] and [v["name"] for v in state["versions"]] == [
        "studio_test_v1",
        "studio_test_v2",
    ]
    assert [item["name"] for item in training.examples(root)["models"]] == ["studio_test_v1", "studio_test_v2"]
    # Every version, and the base, on the same validation tiles.
    assert training.examples(root)["scores"] is None
    scores = training.score_versions(settings, root / "job3", lambda *args: None)
    assert [row["name"] for row in scores["rows"]] == ["nuclei", "studio_test_v1", "studio_test_v2"]
    assert scores["rows"][2]["model"] == second["model"] and len(scores["tiles"]) == 1 and scores["cells"] > 0
    assert all(0 <= row["recall"] <= 1 and row["tp"] + row["fn"] == scores["cells"] for row in scores["rows"])
    assert training.examples(root)["scores"]["current"] is True
    training.set_role(root, scores["tiles"], "train")  # the tiles changed: the table says so
    assert training.examples(root)["scores"]["current"] is False
    with pytest.raises(ValueError, match="validation"):
        training.score_versions(settings, root / "job4", lambda *args: None)
    with pytest.raises(ValueError, match="exist"):
        training.start(root, "studio_test", "nuclei")
