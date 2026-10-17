"""Studio's live pixel classes use Tab 4's feature stack and clustering."""

import base64
import io

import pytest
from PIL import Image

from spacemap_gui.studio import pixelclasses
from test_studio_artifacts import marker_image, slide  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def regions(tmp_path_factory):
    """Tissue in four regions, each lit by its own markers, with smooth noise as in real tissue."""
    import numpy as np
    from scipy import ndimage

    rng = np.random.default_rng(5)
    size = 640
    tissue = np.zeros((size, size), bool)
    tissue[64:576, 64:576] = True
    lit = {"CD3": (slice(64, 320), slice(64, 320)), "PanCK": (slice(64, 320), slice(320, 576))}
    lit["CD68"] = (slice(320, 576), slice(64, 320))
    markers = []
    for name in ("DAPI", "CD3", "PanCK", "CD68", "CD20"):
        plane = ndimage.gaussian_filter(rng.gamma(4.0, 150.0, (size, size)), 2.0) * tissue
        if name in lit:
            plane[lit[name]] += 2500.0
        if name == "DAPI":
            plane += 800.0 * tissue
        markers.append(plane)
    he = [np.where(tissue, 120 + 20 * rng.random((size, size)), 245) for _ in range(3)]
    data = np.clip(np.stack(he + markers), 0, 65535).astype(np.uint16)
    path = tmp_path_factory.mktemp("regions") / "R1_roi_1.ome.tiff"
    marker_image(path, data, names=["HE_Red", "HE_Green", "HE_Blue", "DAPI", "CD3", "PanCK", "CD68", "CD20"])
    return str(path)


def picture(found):
    return Image.open(io.BytesIO(base64.b64decode(found["png"].split(",", 1)[1])))


def test_classes_cover_the_image_and_hiding_only_redraws(slide, monkeypatch):  # noqa: F811
    import spacemap.tissue_annotation_utils as tab4

    found = pixelclasses.classify(slide, {}, 5)
    assert found["n_clusters"] == 5 and len(found["classes"]) == 5 and found["pixels"] > 0
    assert found["image_width"] == 600 and not found["from_saved_tissue"]
    image = picture(found)
    assert image.mode == "RGBA" and image.size[0] * pixelclasses.prepare(slide, {})["factor"] >= 590
    assert sum(item["percent"] for item in found["classes"]) == pytest.approx(100, abs=0.5)
    profile = found["profile"]
    assert profile["rows"] == [str(item["id"]) for item in found["classes"]] and len(profile["columns"]) == 6
    assert len(profile["values"]) == 5 and len(profile["values"][0]) == 6

    def refuse(*args, **kwargs):
        raise AssertionError("Hiding a class must not cluster again")

    monkeypatch.setattr(tab4, "run_pixel_clustering", refuse)
    hidden = pixelclasses.classify(slide, {}, 5, hidden=[found["classes"][0]["id"]])
    assert hidden["png"] != found["png"]
    with pytest.raises(ValueError):
        pixelclasses.classify(slide, {}, 1)


def tab4(**values):
    return {"tab_configurations": {"tab4_annotation": values}}


def test_names_save_into_tab4_domain_settings(tmp_path):
    import json

    from spacemap_gui.studio.settings import read_config, revision

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps(tab4(annot_ta_domain_names=["Stroma"])))
    before = pixelclasses.names(read_config(root), "S1_roi_1", 6)
    assert before["domains"] == {} and before["names"] == ["Stroma"]
    saved = pixelclasses.save_names(
        root, "S1_roi_1", 6, {"0": "Tumor", "2": "Stroma", "3": " ", "4": "Unassigned"}, [5], revision(before)
    )
    stored = json.loads((root / "study.json").read_text())["tab_configurations"]["tab4_annotation"]
    assert stored["annot_ta_domain_map"] == {"0": "Tumor", "2": "Stroma"}
    assert stored["annot_ta_excluded_clusters"] == [5]
    assert stored["annot_ta_domain_names"] == ["Stroma", "Tumor"]
    assert stored["annot_ta_cluster_config"]["n_clusters"] == 6
    assert set(stored["annot_ta_domain_colors"]) == {"Tumor", "Stroma"}
    assert saved["domains"] == {"0": "Tumor", "2": "Stroma"} and saved["excluded"] == [5]
    # Names belong to the image and class count they were made for.
    config = read_config(root)
    assert pixelclasses.names(config, "S1_roi_1", 7)["domains"] == {}
    assert pixelclasses.names(config, "S1_roi_2", 6)["domains"] == {}
    with pytest.raises(ValueError, match="another window"):
        pixelclasses.save_names(root, "S1_roi_1", 6, {}, [], revision(before))
    for domains, excluded in (({"9": "Tumor"}, []), ({"0": "a/b"}, []), ({}, [True]), ({"x": "Tumor"}, [])):
        with pytest.raises(ValueError):
            pixelclasses.check_names(6, domains, excluded)


def test_the_map_is_coloured_by_domain(slide):  # noqa: F811
    classes = pixelclasses.classify(slide, {}, 5)
    found = pixelclasses.classify(slide, {}, 5, domains={"0": "Tumor", "1": "Tumor", "2": "Stroma"}, excluded=[3])
    share = {item["id"]: item["percent"] for item in classes["classes"]}
    legend = {item["name"]: item for item in found["domains"]}
    assert list(legend) == ["Tumor", "Stroma"] and legend["Tumor"]["classes"] == [0, 1]
    assert legend["Tumor"]["percent"] == pytest.approx(share[0] + share[1], abs=0.02)
    import numpy as np

    colours = {tuple(c) for c in np.asarray(picture(found)).reshape(-1, 4) if c[3]}
    wanted = {legend[name]["colour"].lstrip("#") for name in legend} | {"555555"}
    assert {"%02x%02x%02x" % c[:3] for c in colours} <= wanted


def test_export_writes_tab4_files_with_the_preview_classes(regions, tmp_path, monkeypatch):
    import json

    config = tab4(
        annot_ta_domain_map={"0": "Tumor", "1": "Tumor", "2": "Stroma"},
        annot_ta_excluded_clusters=[3],
        annot_ta_domain_map_image={"sample": "R1_roi_1", "n_clusters": 4},
    )
    with pytest.raises(ValueError, match="Save the class names"):
        pixelclasses.export(regions, config, tmp_path / "other", "R1_roi_2")
    # At the preview's own resolution the exported classes are the preview's.
    same = pixelclasses.export(regions, config, tmp_path / "same", "R1_roi_1")
    shown = {item["id"]: item["percent"] for item in pixelclasses.classify(regions, config, 4)["classes"]}
    assert same["classes"] == pytest.approx(shown, abs=0.01)
    names = sorted(path.split("/")[-1] for path in same["files"])
    assert names == [
        "R1_roi_1_macro_domains.geojson",
        "R1_roi_1_raw_macro_clusters.geojson",
        "R1_roi_1_valid_tissue_from_clusters_highres.geojson",
    ]
    domains = json.loads((tmp_path / "same/R1_roi_1_macro_domains.geojson").read_text())["features"]
    assert sorted(f["properties"]["name"] for f in domains) == ["Stroma", "Tumor"]
    # A finer export applies the preview's class centres in the preview's units: the classes keep
    # their markers, and the regions come out at their true size (a quarter of the tissue each).
    monkeypatch.setattr(pixelclasses, "PREVIEW_SIDE", 100)
    reserve = tab4(
        **config["tab_configurations"]["tab4_annotation"],
        annot_ta_cluster_config={"clustering_downsample": 2, "low_signal_handling": "reserve"},
    )
    preview = pixelclasses.classify(regions, reserve, 4)["classes"]
    lit = {item["top_markers"][0]: item["id"] for item in preview if item["top_markers"]}
    fine = pixelclasses.export(regions, reserve, tmp_path / "fine", "R1_roi_1")
    assert fine["factor"] == 2 and pixelclasses.prepare(regions, reserve)["factor"] == 8
    assert {"CD3", "PanCK", "CD68"} <= set(lit)
    for marker in ("CD3", "PanCK", "CD68"):
        assert fine["classes"][lit[marker]] == pytest.approx(25, abs=1), (marker, preview, fine["classes"])
    assert sum(fine["classes"].values()) == pytest.approx(100, abs=0.1)
    assert (tmp_path / "fine/R1_roi_1_macro_domains.geojson").is_file()


def test_the_preview_normalisation_reproduces_tab4_features(slide):  # noqa: F811
    import numpy as np
    from spacemap.tissue_annotation_utils import PyramidImageReader

    prepared = pixelclasses.prepare(slide, {})
    stack, reader = prepared["stack"], PyramidImageReader(slide)
    scale = pixelclasses.scaling(reader, stack, prepared["values"])
    again = pixelclasses.features_like(
        reader, stack, scale, stack["valid_tissue_mask"], prepared["factor"], prepared["values"]
    )
    for key in ("pixels_original_valid", "pixels_z_valid", "features"):
        assert np.array_equal(again[key], stack[key]), key
