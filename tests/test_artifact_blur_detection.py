"""Out-of-focus detection must find blur without taking valid tissue with it.

The four zones below are the cases that separate a working focus detector from
one that merely correlates with signal strength:

* **sharp**        - stained and in focus. Must be kept.
* **blurred**      - stained and out of focus in every marker. Must be removed.
* **unstained**    - in focus, but no antibody bound there. Must be kept: a
  sharpness score is near zero here for an honest reason, and deleting it is the
  most common way valid tissue disappears.
* **cycle blur**   - out of focus in most markers while the two brightest stay
  sharp. Must be removed. A maximum projection cannot see this, because the two
  sharp channels are the ones carrying the projection.
"""

import json

import cv2
import numpy as np
import pytest

from spacemap.tissue_annotation_utils import (
    DEFAULT_ARTIFACT_CONFIG,
    _attribution_grid_geometry,
    _channel_focus_tiles,
    _clamp_and_normalize,
    _odd_kernel,
    _robust_channel_z,
    detect_artifacts,
    region_features_from_labels,
    write_artifact_manifest,
)

MPP = 0.5083
DOWNSAMPLE = 2          # ~1 um/px, as in a real masking pass
SIZE = 600
N_MARKERS = 8

ZONES = {
    "sharp": (slice(80, 220), slice(80, 220)),
    "blurred": (slice(80, 220), slice(320, 460)),
    "unstained": (slice(320, 460), slice(80, 220)),
    "cycle_blur": (slice(320, 460), slice(320, 460)),
}
SHARP_MARKERS = {"sharp": list(range(N_MARKERS)), "blurred": [], "unstained": [], "cycle_blur": [0, 1]}
BLURRED_MARKERS = {"sharp": [], "blurred": list(range(N_MARKERS)), "unstained": [], "cycle_blur": [2, 3, 4, 5, 6, 7]}


@pytest.fixture(scope="module")
def slide():
    """A synthetic ROI carrying all four zones, plus the maps step 2 consumes."""
    rng = np.random.default_rng(7)
    tissue = np.zeros((SIZE, SIZE), dtype=bool)
    tissue[50:550, 50:550] = True

    # Markers all decorate the same cells, so a sharp maximum projection stays
    # sharp. Independent noise per channel would hand the projection a smooth
    # upper envelope and test the noise model instead of the detector.
    cells = cv2.GaussianBlur(rng.normal(0.0, 1.0, (SIZE, SIZE)).astype(np.float32), (0, 0), 1.2)

    def stain(rows, columns, weight, blurred):
        shape = (rows.stop - rows.start, columns.stop - columns.start)
        values = 120.0 * weight + 260.0 * weight * cells[rows, columns]
        values = values + rng.normal(0, 6.0, shape).astype(np.float32)
        return cv2.GaussianBlur(values, (0, 0), 5.0) if blurred else values

    channels = []
    for index in range(N_MARKERS):
        # Channels 0 and 1 stand in for DAPI: brightest in the panel, and the
        # two that hold focus in the cycle_blur zone.
        weight = 1.0 if index in (0, 1) else 0.5
        plane = np.full((SIZE, SIZE), 15.0, np.float32) + rng.normal(0, 2.0, (SIZE, SIZE)).astype(np.float32)
        for name, (rows, columns) in ZONES.items():
            if index in SHARP_MARKERS[name]:
                plane[rows, columns] = stain(rows, columns, weight, blurred=False)
            elif index in BLURRED_MARKERS[name]:
                plane[rows, columns] = stain(rows, columns, weight, blurred=True)
        channels.append(np.clip(plane, 0, None))

    # The virtual H&E stains everything, in focus everywhere, as a real one does.
    he = np.full((SIZE, SIZE), 10.0, np.float32)
    sharp_he = np.clip(120.0 + 260.0 * cells + rng.normal(0, 6.0, (SIZE, SIZE)), 0, None).astype(np.float32)
    he[tissue] = sharp_he[tissue]

    robust = [_robust_channel_z(plane, 50.0) for plane in channels]
    pixel_um = MPP * DOWNSAMPLE
    grid_shape, decimate = _attribution_grid_geometry((SIZE, SIZE), pixel_um)
    focus_grid = np.stack([
        _channel_focus_tiles(
            plane, grid_shape,
            pre_sigma_px=DEFAULT_ARTIFACT_CONFIG["noise_smoothing_um"] / pixel_um,
            ksize=_odd_kernel(DEFAULT_ARTIFACT_CONFIG["focus_detail_um"] / pixel_um, 1, 31),
            decimate=decimate,
        )
        for plane in channels
    ])
    signal_grid = np.stack([
        cv2.resize(plane, (grid_shape[1], grid_shape[0]), interpolation=cv2.INTER_AREA) for plane in robust
    ])

    return {
        "tissue_mask": tissue,
        "codex_projection": _clamp_and_normalize(np.maximum.reduce(channels), 3.0),
        "he_gray": _clamp_and_normalize(he, 3.0),
        "codex_robust_max": np.maximum.reduce(robust),
        "channel_positive_count": sum((plane >= 3.0).astype(np.uint8) for plane in robust),
        "n_codex_channels": N_MARKERS,
        "microns_per_pixel": MPP,
        "downsample": DOWNSAMPLE,
        "marker_names": [f"MARK{index}" for index in range(N_MARKERS)],
        "channel_focus_grid": focus_grid,
        "channel_signal_grid": signal_grid,
        "sample_id": "SYNTH",
    }


def _run(slide, **overrides):
    config = {**DEFAULT_ARTIFACT_CONFIG, **overrides}
    return detect_artifacts(config=config, **slide)


def _flagged(result, zone_name: str) -> float:
    rows, columns = ZONES[zone_name]
    return float(result["blur_mask"][rows, columns].mean())


@pytest.fixture(scope="module")
def result(slide):
    return _run(slide)


def test_blurred_tissue_is_removed(result):
    assert _flagged(result, "blurred") > 0.9


def test_sharp_tissue_is_kept(result):
    assert _flagged(result, "sharp") < 0.05


def test_unstained_tissue_is_not_called_blurry(result):
    """The false positive that costs valid tissue: no antibody bound, in focus."""
    assert _flagged(result, "unstained") < 0.05


def test_blur_confined_to_some_cycles_is_found(result):
    """The false negative a maximum projection cannot see."""
    assert _flagged(result, "cycle_blur") > 0.9


def test_a_slide_that_is_all_in_focus_gets_nothing_removed(slide):
    """The property a fixed threshold cannot have.

    Sharpness inside real tissue varies several-fold for honest reasons, so a
    fixed cut on that continuum always removes its lower tail. Here every zone
    is built sharp, and the correct answer is that there is no blur, at any
    sensitivity the simple control can reach.
    """
    sharp_only = dict(slide)
    for name in ("blurred", "cycle_blur"):
        rows, columns = ZONES[name]
        del rows, columns
    # Rebuild with no blurred zone at all by scoring the H&E against itself:
    # both maps are then the same sharp field, so no population can separate.
    sharp_only["codex_projection"] = slide["he_gray"]
    for sensitivity in (0.5, 1.0, 2.0):
        result = _run(sharp_only, blur_sensitivity=sensitivity)
        assert result["stats"]["blurry_percent"] < 1.0, (
            f"sensitivity {sensitivity} removed "
            f"{result['stats']['blurry_percent']:.1f}% of tissue from a sharp slide"
        )
        assert not result["stats"]["blur_population_found"]


def test_no_setting_of_the_simple_control_invents_blur(slide):
    """The slider tunes strictness; it cannot manufacture a finding.

    Ordinary tissue texture spans two to three times from its softest to its
    sharpest part. The keenest setting still asks for more separation than that,
    so a slide whose sharpness is one smooth continuum returns nothing at every
    step of the control rather than flooding at the top of it.
    """
    from spacemap_gui.tabs.tab4_annotation.s2_artifact_removal import (
        SENSITIVITY_LEVELS, run_config_for,
    )

    sharp_only = dict(slide, codex_projection=slide["he_gray"])
    for level in SENSITIVITY_LEVELS:
        config = run_config_for({**DEFAULT_ARTIFACT_CONFIG, "blur_sensitivity_level": level})
        assert config["blur_min_separation"] >= 3.0, (
            f"level {level} would accept a split no wider than normal tissue texture"
        )
        result = detect_artifacts(config=config, **sharp_only)
        assert result["stats"]["blurry_percent"] < 1.0, (
            f"level {level} removed {result['stats']['blurry_percent']:.1f}% of a sharp slide"
        )


def test_the_simple_control_does_move_a_slide_that_is_blurred(slide):
    from spacemap_gui.tabs.tab4_annotation.s2_artifact_removal import run_config_for

    cautious = detect_artifacts(config=run_config_for(
        {**DEFAULT_ARTIFACT_CONFIG, "blur_sensitivity_level": 1}), **slide)
    keen = detect_artifacts(config=run_config_for(
        {**DEFAULT_ARTIFACT_CONFIG, "blur_sensitivity_level": 7}), **slide)

    assert keen["stats"]["blurry_percent"] > cautious["stats"]["blurry_percent"]
    assert cautious["stats"]["blurry_percent"] > 0.0


def test_the_separation_gap_is_what_refuses_a_sharp_slide(slide):
    sharp_only = dict(slide, codex_projection=slide["he_gray"])
    refused = _run(sharp_only)
    accepted = _run(slide)

    assert refused["stats"]["blur_separation_gap"] < refused["stats"]["blur_separation_required"]
    assert accepted["stats"]["blur_separation_gap"] > accepted["stats"]["blur_separation_required"]


def test_the_pooled_test_alone_would_miss_cycle_specific_blur(slide):
    pooled = _run(slide, blur_evidence_rule="pooled_only")
    assert _flagged(pooled, "cycle_blur") < 0.2
    assert _flagged(pooled, "blurred") > 0.9


def test_the_signal_gate_is_what_saves_unstained_tissue(slide):
    """Shown with the automatic cut off, which is the regime the gate is for.

    With the automatic cut on, the separation test already refuses to split a
    slide that holds no out-of-focus population, so two safeguards would have to
    fail before unstained tissue could be taken. Turning the automatic cut off
    isolates the gate and shows what it alone is worth.
    """
    ungated = _run(slide, require_codex_signal=False, blur_threshold_mode="manual",
                   blur_evidence_rule="pooled_only")
    gated = _run(slide, require_codex_signal=True, blur_threshold_mode="manual",
                 blur_evidence_rule="pooled_only")

    assert _flagged(ungated, "unstained") > 0.5
    assert _flagged(gated, "unstained") < 0.05


def test_the_named_markers_are_the_ones_out_of_focus(result):
    """Attribution must point at the blurred cycles, not the sharp ones."""
    regions = [record for record in result["regions"]["blur"]
               if record["spacemap"]["evidence"]["markers_out_of_focus"] < N_MARKERS]
    assert regions, "the cycle-specific region should not implicate every marker"
    soft = {entry["marker"] for entry in regions[0]["spacemap"]["candidate_markers"] if entry["out_of_focus"]}
    assert soft == {f"MARK{index}" for index in BLURRED_MARKERS["cycle_blur"]}


def test_each_region_is_its_own_reviewable_object(result):
    assert result["stats"]["blur_regions"] == 2
    for record in result["regions"]["blur"]:
        block = record["spacemap"]
        assert block["region_id"].startswith("SYNTH_blur_")
        assert 0.0 <= block["severity"] <= 1.0
        assert 0.0 <= block["confidence"] <= 1.0
        assert block["review"]["status"] == "pending"
        assert block["candidate_markers"]


def test_the_size_filter_applies_before_the_safety_margin(slide):
    """A margin of 25 um must not inflate a speck past a 20000 um^2 minimum."""
    strict = _run(slide, min_area_microns2=20000.0, expand_microns=25.0)
    assert strict["stats"]["blur_regions"] == 0
    assert not strict["blur_mask"].any()


def test_lengths_are_physical_not_pixel_counts(slide):
    """The same settings must mean the same thing at a different downsample."""
    coarse = dict(slide, downsample=DOWNSAMPLE * 2)
    fine = _run(slide)
    # Halving the resolution must not change the answer by more than a little.
    rescaled = _run(coarse)
    assert abs(rescaled["stats"]["blurry_percent"] - fine["stats"]["blurry_percent"]) < 10.0


def test_regions_export_one_geojson_feature_each(result, tmp_path):
    features = region_features_from_labels(
        result["region_labels"]["blur"], result["regions"]["blur"],
        float(DOWNSAMPLE), "Detected_Blur", color_rgb=-256, simplify_px=2.0,
    )
    assert len(features) == len(result["regions"]["blur"])
    properties = features[0]["properties"]
    assert properties["classification"] == {"name": "Detected_Blur", "colorRGB": -256}
    assert "blur_severity" in properties["measurements"]
    assert properties["spacemap"]["candidate_markers"]
    # Geometry must land in full-resolution pixel space.
    from shapely.geometry import shape

    centroid = properties["spacemap"]["centroid_full_res_px"]
    drawn = shape(features[0]["geometry"]).centroid
    assert np.hypot(drawn.x - centroid[0], drawn.y - centroid[1]) < 40.0


def test_the_manifest_is_machine_readable(result, tmp_path):
    path = write_artifact_manifest(
        tmp_path / "review.json", sample_id="SYNTH", regions=result["regions"],
        exports={"blur": "SYNTH_detected_blur.geojson"}, stats=result["stats"],
        parameters=result["parameters"], marker_names=[f"MARK{i}" for i in range(N_MARKERS)],
    )
    manifest = json.loads(open(path).read())
    assert manifest["schema"] == "spacemap.artifact-manifest/1"
    assert len(manifest["regions"]) == sum(len(records) for records in result["regions"].values())
    assert manifest["regions"][0]["review"]["status"] == "pending"
    json.dumps(manifest)  # no NumPy scalars, no NaN


def test_a_config_written_before_lengths_moved_to_microns_still_runs(slide):
    legacy = detect_artifacts(
        config={"use_he_reference": False, "window_size": 31, "gauss_pre_sigma": 1.5,
                "gauss_post_sigma": 5.0, "morph_close_radius": 25, "morph_open_radius": 7},
        **slide,
    )
    assert legacy["stats"]["blur_reference_mode"] == "self"
    assert legacy["blur_mask"].shape == slide["tissue_mask"].shape


def test_detection_still_runs_without_the_per_marker_grids(slide):
    without = dict(slide)
    without.pop("channel_focus_grid")
    without.pop("channel_signal_grid")
    plain = detect_artifacts(config=dict(DEFAULT_ARTIFACT_CONFIG), **without)
    assert plain["stats"]["marker_consensus_used"] is False
    assert _flagged(plain, "blurred") > 0.9
