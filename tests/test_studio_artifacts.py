"""Studio's live artifact sliders give exactly what Tab 4's detectors give."""

import json

import numpy as np
import pytest
from scipy import ndimage

from spacemap_gui.studio import artifacts


def marker_image(path, data, names):
    """A pyramidal OME-TIFF whose channel names Tab 4's reader can see (it needs the OME namespace)."""
    import tifffile

    count, height, width = data.shape
    xml = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0">'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="{width}" SizeY="{height}" '
        f'SizeC="{count}" SizeZ="1" SizeT="1">'
        + "".join(f'<Channel ID="Channel:0:{i}" Name="{name}"/>' for i, name in enumerate(names))
        + "</Pixels></Image></OME>"
    )
    with tifffile.TiffWriter(path) as writer:
        for index, channel in enumerate(data):
            writer.write(channel, subifds=1, tile=(16, 16), metadata=None, description=xml if index == 0 else None)
            writer.write(channel[::2, ::2], subfiletype=1, tile=(16, 16), metadata=None)


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    """Textured tissue with a blurred patch, a flat bright patch, and a lint fibre on the glass."""
    rng = np.random.default_rng(3)
    size = 600
    tissue = np.zeros((size, size), bool)
    tissue[80:520, 80:520] = True
    markers = []
    for index in range(6):
        texture = rng.gamma(2.0, 400.0, (size, size)) * tissue
        texture = ndimage.gaussian_filter(texture, 1.0)
        texture[300:450, 300:450] = ndimage.gaussian_filter(texture, 8.0)[300:450, 300:450]  # out of focus
        texture[100:260, 100:260] = 6000.0  # flat glare lights every marker, 80 um across
        texture[36:56, 100:400] = 9000.0  # a lint fibre on the glass, 10 um wide (thinner reads as a scanner seam)
        markers.append(texture)
    he = [np.where(tissue, 120 + 20 * rng.random((size, size)), 245) for _ in range(3)]
    data = np.clip(np.stack(he + markers), 0, 65535).astype(np.uint16)
    path = tmp_path_factory.mktemp("slide") / "S1_roi_1.ome.tiff"
    marker_image(path, data, names=["HE_Red", "HE_Green", "HE_Blue", "DAPI", "CD3", "CD8", "CD20", "PanCK", "CD68"])
    return str(path)


@pytest.fixture(scope="module")
def prepared(slide):
    return artifacts.prepare(slide, {"tab_configurations": {"tab4_annotation": {"annot_ta_artifact_markers": []}}})


def tab4(prepared, **config):
    from spacemap.tissue_annotation_utils import detect_artifacts

    base = {
        **prepared["settings"],
        "write_review_manifest": False,
        "detect_bright_artifacts": False,
        "detect_filamentous_lint": False,
        "blur_threshold_mode": "manual",
        "blur_ratio_threshold": 0.0,
        "blur_evidence_rule": "pooled_only",
    }
    return detect_artifacts(config={**base, **config}, **prepared["inputs"])


@pytest.mark.parametrize("cut", [0.2, 0.45, 0.8])
def test_live_blur_equals_tab4(prepared, cut):
    expected = tab4(prepared, blur_ratio_threshold=cut)["blur_mask"]
    assert np.array_equal(artifacts.blur_mask(prepared, cut), expected)


@pytest.mark.parametrize("percentile", [95.0, 99.5])
def test_live_bright_equals_tab4(prepared, percentile):
    expected = tab4(prepared, detect_bright_artifacts=True, bright_flat_mean_percentile=percentile)["bright_mask"]
    assert np.array_equal(artifacts.bright_mask(prepared, percentile), expected)


@pytest.mark.parametrize("z", [4.0, 12.0])
def test_live_lint_equals_tab4(prepared, z):
    expected = tab4(prepared, detect_filamentous_lint=True, lint_robust_z_threshold=z)["lint_mask"]
    assert np.array_equal(artifacts.lint_mask(prepared, z), expected)


def test_sliders_find_what_was_made(prepared, slide):
    scale = prepared["downsample"]
    # Where blur should be is judged on real slides; here the live masks must equal Tab 4's (above).
    assert artifacts.blur_mask(prepared, 0.0).sum() == 0, "A zero cut switches blur off"
    assert artifacts.lint_mask(prepared, 6.0)[int(46 / scale), int(250 / scale)], "The fibre on the glass is lint"
    preview = artifacts.preview(slide, {}, "blur", 0.6)
    assert 0 <= preview["percent"] <= 100 and preview["regions"] >= 0 and preview["colour"] == artifacts.COLOURS["blur"]


def test_studio_lint_can_search_on_tissue_without_changing_classic(prepared):
    # Treat the known fluorescent fibre as lying within the assigned Tissue.
    domain = np.ones(prepared['shape'], dtype=bool)
    assigned = {**prepared, 'inputs': {**prepared['inputs'], 'tissue_mask': domain}}
    scale = prepared['downsample']
    assert not artifacts.lint_mask(assigned, 6).any()
    assert artifacts.lint_mask(assigned, 6, domain=domain)[int(46 / scale), int(250 / scale)]


def test_sliders_save_into_tab4_settings(tmp_path):
    from spacemap_gui.studio.settings import read_config, revision

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({"tab_configurations": {"tab4_annotation": {}}}))
    current = artifacts.values(read_config(root))
    assert current["blur"]["auto"] and current["clot"]["on"] is False
    saved = artifacts.save(
        root, {"blur": {"value": 0.4}, "bright": {"value": 98.0, "on": False}, "clot": {"on": True}}, revision(current)
    )
    stored = json.loads((root / "study.json").read_text())["tab_configurations"]["tab4_annotation"]
    assert stored["annot_ta_artifact_config"]["blur_ratio_threshold"] == 0.4
    assert stored["annot_ta_artifact_config"]["blur_threshold_mode"] == "manual"
    assert stored["annot_ta_artifact_config"]["blur_evidence_rule"] == "pooled_only"
    assert stored["annot_ta_artifact_config"]["detect_bright_artifacts"] is False
    assert stored["annot_ta_tissue_config"]["detect_blood_clot"] is True
    assert saved["blur"] == {"value": 0.4, "on": True, "auto": False}
    with pytest.raises(ValueError):
        artifacts.check({"blur": {"value": 7}})
    with pytest.raises(ValueError):
        artifacts.check({"smudge": {"value": 1}})


def test_apply_writes_tab4_files(slide, tmp_path):
    written = artifacts.apply(slide, {}, tmp_path / "stage0", "S1_roi_1")["written"]
    names = sorted(p.split("/")[-1] for p in written)
    assert "S1_roi_1_valid_tissue.geojson" in names and "S1_roi_1_detected_blur.geojson" in names
    assert "S1_roi_1_tissue_boundary.geojson" in names and "S1_roi_1_detected_artifacts.geojson" in names
    assert (tmp_path / "stage0/S1_roi_1_valid_tissue.geojson").is_file()
    assert (tmp_path / "stage0/intermediate/S1_roi_1_detected_lint.geojson").is_file()


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_an_example_sets_the_slider_to_catch_it(slide):
    bright = artifacts.suggest(slide, {}, "bright", [box(130, 130, 230, 230)])
    # Tab 4 also needs glare to be textureless, which the patch border is not.
    assert 90 <= bright["value"] <= 99.99 and bright["caught"] >= 0.5, bright
    lint = artifacts.suggest(slide, {}, "lint", [box(150, 38, 350, 54)])
    assert lint["caught"] >= 0.5 and lint["value"] <= 40, lint
    with pytest.raises(ValueError, match="glass"):
        artifacts.suggest(slide, {}, "lint", [box(250, 250, 300, 300)])
    with pytest.raises(ValueError, match="tissue"):
        artifacts.suggest(slide, {}, "bright", [box(2, 2, 30, 30)])


def test_a_clean_example_moves_the_cut_between(slide):
    alone = artifacts.suggest(slide, {}, "bright", [box(130, 130, 230, 230)])
    both = artifacts.suggest(slide, {}, "bright", [box(130, 130, 230, 230)], clean=[box(200, 300, 280, 380)])
    assert both["value"] <= alone["value"]
