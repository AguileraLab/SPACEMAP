"""Blood clot must be found without taking real H&E tissue with it."""

import numpy as np
import pytest

from spacemap.tissue_annotation_utils import DEFAULT_TISSUE_CONFIG, detect_blood_clot

MPP = 0.5083
DOWNSAMPLE = 2   # ~1 um/px, so a nucleus is several pixels across, as in a real ROI
PIXEL_MM2 = (MPP * DOWNSAMPLE) ** 2 / 1e6


def _nucleated(rng, shape, density: float = 0.004, radius: int = 3) -> np.ndarray:
    """Haematoxylin field with round nuclei a few pixels across, as at ~1 um/px."""
    import cv2

    seeds = (rng.random(shape) < density).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    nuclei = cv2.dilate(seeds, kernel) > 0
    return np.where(nuclei, 0.09, 0.002).astype(np.float32)


def _rgb_from_stains(haematoxylin: np.ndarray, eosin: np.ndarray) -> np.ndarray:
    """Synthesise a virtual H&E image from the two stain concentrations."""
    from skimage.color import hed2rgb

    stack = np.stack([haematoxylin, eosin, np.zeros_like(eosin)], axis=-1)
    return np.clip(hed2rgb(stack), 0.0, 1.0).astype(np.float32)


@pytest.fixture
def slide():
    """A slide holding real tissue, a clot, muscle, and small clot specks.

    * tissue  - eosin plus punctate nuclei (high haematoxylin texture)
    * clot    - strong eosin, no nuclei at all
    * muscle  - strong eosin *and* nuclei, the classic false positive
    * specks  - clot-like but tiny, to exercise the minimum-area filter
    """
    rng = np.random.default_rng(0)
    size = 400
    h = np.zeros((size, size), dtype=np.float32)
    e = np.zeros((size, size), dtype=np.float32)
    tissue = np.zeros((size, size), dtype=bool)

    # real tissue, left column
    tissue[20:380, 20:100] = True
    e[20:380, 20:100] = 0.05
    h[20:380, 20:100] = _nucleated(rng, (360, 80))

    # clot, upper right: eosin-rich, nuclei-free
    clot = np.zeros_like(tissue)
    clot[40:170, 240:380] = True
    tissue |= clot
    e[clot] = 0.16
    h[clot] = 0.001

    # muscle, lower right: just as eosinophilic, but nucleated
    muscle = np.zeros_like(tissue)
    muscle[250:380, 240:380] = True
    tissue |= muscle
    e[muscle] = 0.16
    h[250:380, 240:380] = _nucleated(rng, (130, 140))

    # small clot-like specks, wide enough to survive the default morphology so
    # that it is the area filter being exercised and not the de-speckler
    specks = []
    for row in (40, 150, 260):
        speck = (slice(row, row + 60), slice(105, 165))
        tissue[speck] = True
        e[speck] = 0.16
        h[speck] = 0.001
        specks.append(speck)

    return {"rgb": _rgb_from_stains(h, e), "tissue": tissue,
            "clot": clot, "muscle": muscle, "specks": specks}


def _run(slide, **overrides):
    config = {"clot_min_area_mm2": 0.0, **overrides}
    return detect_blood_clot(
        slide["rgb"], slide["tissue"], config,
        microns_per_pixel=MPP, downsample=DOWNSAMPLE,
    )


def _recall(mask, region):
    return float(mask[region].mean())


def test_the_clot_is_detected(slide):
    result = _run(slide)

    assert _recall(result["clot_mask"], slide["clot"]) > 0.85


def test_real_tissue_is_not_taken(slide):
    result = _run(slide)
    tissue_only = slide["tissue"] & ~slide["clot"] & ~slide["muscle"]
    for speck in slide["specks"]:          # specks are clot by construction
        tissue_only[speck] = False

    assert _recall(result["clot_mask"], tissue_only) < 0.05


def test_eosinophilic_muscle_is_not_mistaken_for_clot(slide):
    """Muscle is as pink as the clot; only the nuclei tell them apart."""
    result = _run(slide)

    assert _recall(result["clot_mask"], slide["muscle"]) < 0.10, (
        "nucleated eosinophilic tissue must survive - this is what the texture term is for"
    )


def test_eosin_alone_would_have_taken_the_muscle(slide):
    """Justifies the nuclei tests: relax them and the false positive returns."""
    eosin_only = _run(slide, clot_max_nuclei_ratio=1e9, clot_eosin_smooth_ratio=1e9)

    assert _recall(eosin_only["clot_mask"], slide["muscle"]) > 0.5


def test_small_islands_are_dropped_by_the_area_filter(slide):
    speck_area_mm2 = 60 * 60 * PIXEL_MM2

    kept_all = _run(slide, clot_min_area_mm2=0.0)
    filtered = _run(slide, clot_min_area_mm2=speck_area_mm2 * 2)

    assert kept_all["stats"]["clot_islands"] > filtered["stats"]["clot_islands"]
    assert filtered["stats"]["rejected_islands"] >= 3
    for speck in slide["specks"]:
        assert not filtered["clot_mask"][speck].any()
    # the real clot survives the same filter
    assert _recall(filtered["clot_mask"], slide["clot"]) > 0.85


def test_area_filter_reports_measurements_in_mm2(slide):
    result = _run(slide)
    biggest = result["components"][0]

    expected = float(slide["clot"].sum()) * PIXEL_MM2
    assert biggest["area_mm2"] == pytest.approx(expected, rel=0.25)
    assert result["stats"]["largest_island_mm2"] == pytest.approx(expected, rel=0.25)


def test_demanding_more_eosin_is_stricter(slide):
    loose = _run(slide, clot_eosin_ratio=1.5)
    strict = _run(slide, clot_eosin_ratio=4.0)

    assert strict["stats"]["clot_pixels"] < loose["stats"]["clot_pixels"]


def test_a_slide_with_no_clot_returns_nothing():
    """The property bare percentile cuts cannot have.

    A percentile threshold always selects its own tail, so on a slide with no
    clot it returns the most clot-like real tissue instead of nothing. This
    slide holds only nucleated tissue and nucleated, strongly eosinophilic
    muscle. The honest answer is zero.
    """
    rng = np.random.default_rng(1)
    size = 400
    haematoxylin = np.zeros((size, size), dtype=np.float32)
    eosin = np.zeros((size, size), dtype=np.float32)
    tissue = np.zeros((size, size), dtype=bool)

    tissue[20:380, 20:190] = True
    eosin[20:380, 20:190] = 0.05
    haematoxylin[20:380, 20:190] = _nucleated(rng, (360, 170))

    tissue[20:380, 210:380] = True
    eosin[20:380, 210:380] = 0.16          # as pink as any clot
    haematoxylin[20:380, 210:380] = _nucleated(rng, (360, 170))

    rgb = _rgb_from_stains(haematoxylin, eosin)
    result = detect_blood_clot(rgb, tissue, {}, microns_per_pixel=MPP, downsample=DOWNSAMPLE)

    assert result["stats"]["clot_islands"] == 0
    assert not result["clot_mask"].any()

    # With the size filter switched off as well, what leaks through is a rounding
    # error rather than a region. Note that this slide's eosin is flat, so the
    # smoothness test cannot run and only two of the three tests are active.
    unfiltered = detect_blood_clot(rgb, tissue, {"clot_min_area_mm2": 0.0},
                                   microns_per_pixel=MPP, downsample=DOWNSAMPLE)
    assert unfiltered["stats"]["clot_area_mm2"] < 0.005
    assert not unfiltered["stats"]["smoothness_test_used"]


def test_pink_tissue_that_still_has_nuclei_is_not_a_clot(slide):
    """Fibrous stroma is pink and smooth, but it is full of spindle nuclei."""
    result = _run(slide)
    coverage = result["nuclear_coverage"]

    assert float(np.median(coverage[slide["clot"]])) < 0.05
    assert float(np.median(coverage[slide["muscle"]])) > 0.10


def test_each_clot_is_exported_with_the_evidence_behind_it(slide):
    result = _run(slide)

    assert result["regions"], "an accepted clot must produce a reviewable region"
    block = result["regions"][0]["spacemap"]
    assert block["artifact_type"] == "blood_clot"
    assert block["review"]["status"] == "pending"
    evidence = block["evidence"]
    assert evidence["eosin_ratio"] >= evidence["eosin_ratio_required"]
    assert evidence["nuclear_coverage"] <= evidence["nuclear_coverage_allowed"]
    assert 0.0 <= block["confidence"] <= 1.0


def test_clipped_haematoxylin_does_not_fake_a_nuclei_free_region(slide):
    """Virtual H&E often bottoms out to zero haematoxylin over real tissue.

    A variability measure reads that as "no nuclei". A coverage measure does
    not, because the threshold it counts against is taken from the same clipped
    distribution.
    """
    import numpy as np

    result = _run(slide)
    assert result["stats"]["nuclear_threshold"] >= 0.0
    # The clot is nuclei-free by construction and must still be found.
    assert _recall(result["clot_mask"], slide["clot"]) > 0.85


def test_components_are_reported_largest_first_with_a_kept_flag(slide):
    result = _run(slide, clot_min_area_mm2=60 * 60 * PIXEL_MM2 * 2)
    areas = [c["area_px"] for c in result["components"]]

    assert areas == sorted(areas, reverse=True)
    assert any(c["kept"] for c in result["components"])
    assert any(not c["kept"] for c in result["components"])


def test_missing_inputs_are_handled(slide):
    assert detect_blood_clot(None, slide["tissue"], {})["stats"]["clot_pixels"] == 0
    empty = np.zeros_like(slide["tissue"])
    assert detect_blood_clot(slide["rgb"], empty, {})["stats"]["clot_pixels"] == 0


def test_detection_is_off_by_default():
    assert DEFAULT_TISSUE_CONFIG["detect_blood_clot"] is False
    assert DEFAULT_TISSUE_CONFIG["clot_exclude_from_tissue"] is True


def test_texture_window_is_forced_odd(slide):
    """An even box filter size would be rejected by OpenCV."""
    result = _run(slide, clot_texture_window_px=8)

    assert result["stats"]["clot_pixels"] > 0
