import json
from pathlib import Path

import pytest
from PIL import Image

import spacemap_assistant.preview_action as preview_action
from spacemap_assistant.paths import ASSISTANT_PROJECTS_DIR
from spacemap_assistant.preview_action import (
    HE_CHANNELS,
    PreviewActionError,
    generate_he_previews,
    generate_non_he_previews,
    generate_overlay_previews,
    generate_scale_bar_previews,
    match_requested_channels,
    parse_he_preview_request,
    parse_visual_assessment_request,
)


def test_parse_he_thumbnail_request_from_roi_folder() -> None:
    folder = ASSISTANT_PROJECTS_DIR / "AI_test" / "outputs" / "rois"
    prompt = f"Display a thumbnail of each image in {folder}, with HE layers only"
    assert parse_he_preview_request(prompt) == {
        "source_path": str(folder),
        "geojson_paths": [],
        "modes": ["he"],
        "request_text": prompt,
        "max_size": 768,
        "scale_bar": None,
    }


def test_parse_he_thumbnail_request_accepts_escaped_underscores() -> None:
    folder = str(ASSISTANT_PROJECTS_DIR / "AI_test" / "outputs" / "rois").replace("_", "\\_")
    parsed = parse_he_preview_request(
        f"Show previews from {folder}. HE layers are HE_Red, HE_Green, HE_Blue"
    )
    assert parsed is not None
    assert parsed["source_path"] == str(ASSISTANT_PROJECTS_DIR / "AI_test" / "outputs" / "rois")


def test_parse_exact_single_image_he_thumbnail_request() -> None:
    image = ASSISTANT_PROJECTS_DIR / "AI_test" / "outputs" / "AI_registered" / "P11_merged.ome.tiff"
    parsed = parse_he_preview_request(f"Now display the HE thumbnail of this image {image}")
    assert parsed is not None
    assert parsed["source_path"] == str(image)


def test_pathless_named_channel_followup_is_a_preview_request() -> None:
    prompt = "Can you display the DAPI layer as a thumbnail?"
    parsed = parse_he_preview_request(prompt)
    assert parsed == {
        "source_path": "",
        "geojson_paths": [],
        "modes": [],
        "request_text": prompt,
        "max_size": 768,
        "scale_bar": None,
    }


def test_scale_bar_followup_keeps_edit_settings_separate_from_channels() -> None:
    prompt = (
        "Can you add the scale bar to the image as a bright white line on the left bottom side "
        "of the same image generated from the previous prompt? "
        "Each pixel is 0.5083 micrometer."
    )
    parsed = parse_he_preview_request(prompt)
    assert parsed is not None
    assert parsed["source_path"] == ""
    assert parsed["modes"] == []
    assert parsed["scale_bar"] == {
        "mpp": 0.5083,
        "length_um": None,
        "position": "bottom-left",
        "color": "#FFFFFF",
    }


def test_combined_he_and_non_he_request_keeps_both_modes() -> None:
    image = ASSISTANT_PROJECTS_DIR / "AI_test" / "registered.ome.tiff"
    parsed = parse_he_preview_request(
        f"Display {image} using HE layers and another thumbnail using non-HE layers"
    )
    assert parsed is not None
    assert parsed["modes"] == ["he", "non_he"]


def test_overlay_request_separates_tiff_and_geojson_paths() -> None:
    image = ASSISTANT_PROJECTS_DIR / "AI_test" / "registered.ome.tiff"
    cells = ASSISTANT_PROJECTS_DIR / "AI_test" / "cells.geojson"
    parsed = parse_he_preview_request(
        f"Overlay the segmentation {cells} on the image {image}"
    )
    assert parsed is not None
    assert parsed["source_path"] == str(image)
    assert parsed["geojson_paths"] == [str(cells)]


def test_named_channels_are_discovered_from_image_metadata_not_a_fixed_list() -> None:
    channels = ["DAPI", "CD3e", "Pan-Cytokeratin", "HE_Red", "HE_Green", "HE_Blue"]
    assert match_requested_channels("Show a DAPI thumbnail", channels) == ["DAPI"]
    assert match_requested_channels("Display CD3e and Pan Cytokeratin layers", channels) == [
        "CD3e",
        "Pan-Cytokeratin",
    ]


def test_non_preview_prompt_is_not_intercepted() -> None:
    assert parse_he_preview_request("How does H&E registration work?") is None
    assert parse_he_preview_request(
        "In this tab, what is the difference between Domain GeoJSON directory "
        "(input for pooling) and Tissue annotation output directory?"
    ) is None


def test_blur_optimization_workflow_is_not_intercepted_as_a_thumbnail() -> None:
    image = ASSISTANT_PROJECTS_DIR / "study" / "outputs" / "rois" / "sample.ome.tiff"
    output = ASSISTANT_PROJECTS_DIR / "study" / "outputs" / "processed_annotations" / "test"
    prompt = f"""
    Run a complete blur-detection optimization and review for this image: {image}
    Save results in {output}. Test and compare several parameter settings, then
    review every detected polygon as a true positive, false positive, or uncertain.
    Generate H&E and DAPI diagnostic thumbnails with overlays and a scale bar.
    Correct the final GeoJSON after making a backup.
    """
    assert parse_he_preview_request(prompt) is None


def test_direct_detected_blur_overlay_remains_a_preview_request() -> None:
    image = ASSISTANT_PROJECTS_DIR / "study" / "outputs" / "rois" / "sample.ome.tiff"
    blur = ASSISTANT_PROJECTS_DIR / "study" / "outputs" / "annotations" / "sample_detected_blur.geojson"
    parsed = parse_he_preview_request(f"Show the detected blur overlay {blur} on image {image}")
    assert parsed is not None
    assert parsed["source_path"] == str(image)
    assert parsed["geojson_paths"] == [str(blur)]


def test_visual_qc_request_finds_the_image_and_geojson() -> None:
    image = ASSISTANT_PROJECTS_DIR / "study" / "outputs" / "rois" / "sample.ome.tiff"
    clot = ASSISTANT_PROJECTS_DIR / "study" / "outputs" / "annotations" / "sample_blood_clot.geojson"
    parsed = parse_visual_assessment_request(
        f"I ran Detect Blood Clot on image ROI {image}. The output is {clot}. "
        "Can you go and check it for me?"
    )
    assert parsed is not None
    assert parsed["source_path"] == str(image)
    assert parsed["geojson_paths"] == [str(clot)]
    assert parsed["preferred_mode"] == "he"


def test_generate_he_previews_reuses_visualization_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "outputs" / "rois"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / "sample.ome.tiff"
    source.write_bytes(b"not decoded because image helpers are mocked")

    import spacemap_gui.tabs.tab9_visualization.overlay_core as overlay_core

    monkeypatch.setattr(
        preview_action,
        "require_assistant_read_path",
        lambda path, **_kwargs: Path(path),
    )
    monkeypatch.setattr(
        preview_action,
        "require_assistant_write_path",
        lambda path: Path(path),
    )
    calls: list[dict[str, object]] = []

    def fake_load(path: Path, **kwargs: object):
        calls.append({"path": path, **kwargs})
        return Image.new("RGB", (64, 32), (150, 90, 70)), (3_200, 6_400), 3

    monkeypatch.setattr(overlay_core, "load_base_image", fake_load)
    results = generate_he_previews(source_dir, max_size=512)

    assert len(results) == 1
    assert Path(results[0]["preview_path"]).is_file()
    assert results[0]["channels"] == list(HE_CHANNELS)
    assert results[0]["pyramid_level"] == 3
    assert results[0]["cached"] is False
    assert calls == [
        {"path": source, "channels": None, "max_size": 512, "require_he": True}
    ]

    calls.clear()
    cached = generate_he_previews(source_dir, max_size=512)
    assert cached[0]["cached"] is True
    assert calls == []


def test_generate_he_previews_rejects_missing_he_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "outputs" / "rois"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "sample.ome.tiff").write_bytes(b"placeholder")

    import spacemap_gui.tabs.tab9_visualization.overlay_core as overlay_core

    monkeypatch.setattr(
        preview_action,
        "require_assistant_read_path",
        lambda path, **_kwargs: Path(path),
    )
    monkeypatch.setattr(
        preview_action,
        "require_assistant_write_path",
        lambda path: Path(path),
    )

    def missing_he(_path: Path, **_kwargs: object):
        raise ValueError("sample.ome.tiff does not contain the required H&E channels")

    monkeypatch.setattr(overlay_core, "load_base_image", missing_he)
    with pytest.raises(PreviewActionError, match="does not contain"):
        generate_he_previews(source_dir)


def test_general_overlay_renderer_draws_cells_and_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preview_action,
        "require_assistant_read_path",
        lambda path, **_kwargs: Path(path),
    )
    monkeypatch.setattr(
        preview_action,
        "require_assistant_write_path",
        lambda path: Path(path),
    )
    base = tmp_path / "base.png"
    Image.new("RGB", (100, 100), (0, 0, 0)).save(base)
    polygon = {
        "type": "Polygon",
        "coordinates": [[[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]]],
    }
    cells = tmp_path / "cell_segmentation.geojson"
    tissue = tmp_path / "tissue_annotations.geojson"
    cells.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": polygon, "properties": {"label": 1}}],
            }
        ),
        encoding="utf-8",
    )
    tissue.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": polygon,
                        "properties": {"classification": {"name": "Tumor", "color": [0, 255, 0]}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    results = generate_overlay_previews(
        [
            {
                "source_path": str(tmp_path / "source.ome.tiff"),
                "preview_path": str(base),
                "preview_mode": "markers",
                "channels": ["DAPI"],
                "pyramid_level": 2,
                "full_height": 100,
                "full_width": 100,
                "preview_width": 100,
                "preview_height": 100,
                "cached": False,
            }
        ],
        [str(cells), str(tissue)],
        prompt="Overlay cell segmentation and tissue annotations",
    )
    assert Path(results[0]["preview_path"]).is_file()
    assert results[0]["overlay_stats"]["cell_polygons"] == 1
    assert results[0]["overlay_stats"]["tissue_polygons"] == 1


def test_scale_bar_edits_the_prior_visual_without_changing_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preview_action,
        "require_assistant_read_path",
        lambda path, **_kwargs: Path(path),
    )
    monkeypatch.setattr(
        preview_action,
        "require_assistant_write_path",
        lambda path: Path(path),
    )
    source = tmp_path / "source.ome.tiff"
    source.write_bytes(b"source placeholder")
    base = tmp_path / "DAPI_CD45_CD68_CD38.png"
    Image.new("RGB", (200, 100), (20, 20, 20)).save(base)
    prior = {
        "source_path": str(source),
        "preview_path": str(base),
        "preview_mode": "markers",
        "channels": ["DAPI", "CD45", "CD68", "CD38"],
        "pyramid_level": 3,
        "full_height": 1_000,
        "full_width": 2_000,
        "preview_width": 200,
        "preview_height": 100,
        "cached": False,
    }
    result = generate_scale_bar_previews(
        [prior],
        mpp=0.5083,
        length_um=100,
        position="bottom-left",
    )[0]
    assert result["channels"] == ["DAPI", "CD45", "CD68", "CD38"]
    assert result["preview_mode"] == "markers"
    assert result["scale_bar"]["source_mpp"] == 0.5083
    assert result["scale_bar"]["length_um"] == 100
    assert Path(result["preview_path"]).is_file()


def test_scale_bar_uses_candidate_crop_width_not_whole_slide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preview_action, "require_assistant_read_path", lambda path, **_kwargs: Path(path))
    monkeypatch.setattr(preview_action, "require_assistant_write_path", lambda path: Path(path))
    source = tmp_path / "source.ome.tiff"
    source.write_bytes(b"source placeholder")
    base = tmp_path / "tile.png"
    Image.new("RGB", (200, 200), (20, 20, 20)).save(base)
    prior = {
        "source_path": str(source),
        "preview_path": str(base),
        "full_width": 2_000,
        "full_height": 2_000,
        "crop_box": [500.0, 500.0, 900.0, 900.0],
    }

    result = generate_scale_bar_previews([prior], mpp=0.5)[0]

    assert result["scale_bar"]["preview_mpp"] == pytest.approx(1.0)
    assert result["scale_bar"]["represented_source_width_px"] == 400.0
    assert result["scale_bar"]["length_um"] == 25.0
