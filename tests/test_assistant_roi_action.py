import json
from pathlib import Path

import pytest

from spacemap_assistant.paths import ASSISTANT_PROJECTS_DIR
from spacemap_assistant.roi_action import (
    ROIActionError,
    _inspect_geojson,
    _validate_inputs,
    parse_roi_extraction_request,
)


AI_TEST = ASSISTANT_PROJECTS_DIR / "AI_test"
IMAGE = AI_TEST / "outputs" / "AI_registered" / "P11_merged.ome.tiff"
BOUNDARIES = AI_TEST / "data" / "P11_roi_bounding_boxes_coords.geojson"
OUTPUT = AI_TEST / "outputs" / "rois"


def test_natural_language_roi_request_extracts_exact_paths() -> None:
    prompt = f"Extract ROIs from {IMAGE} using {BOUNDARIES} and save in {OUTPUT}"
    assert parse_roi_extraction_request(prompt) == {
        "image_path": str(IMAGE),
        "boundaries_path": str(BOUNDARIES),
        "output_dir": str(OUTPUT),
        "overwrite": False,
    }


def test_roi_request_records_explicit_overwrite() -> None:
    prompt = f"Extract ROIs from {IMAGE} using {BOUNDARIES}; save in {OUTPUT} and overwrite existing"
    parsed = parse_roi_extraction_request(prompt)
    assert parsed is not None
    assert parsed["overwrite"] is True


def test_roi_parser_removes_question_mark_after_geojson_path() -> None:
    prompt = f"Extract ROIs from {IMAGE} using {BOUNDARIES}? Save in {OUTPUT} and overwrite"
    parsed = parse_roi_extraction_request(prompt)
    assert parsed is not None
    assert parsed["boundaries_path"] == str(BOUNDARIES)


def test_a_symlinked_alias_stays_in_one_project(tmp_path: Path) -> None:
    """The same alias check as the registration action, created locally."""
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(ASSISTANT_PROJECTS_DIR, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("This filesystem does not support symbolic links.")
    work_image = alias / IMAGE.relative_to(ASSISTANT_PROJECTS_DIR)
    image, boundaries, output, project = _validate_inputs(work_image, BOUNDARIES, OUTPUT)
    assert image == IMAGE.resolve()
    assert boundaries == BOUNDARIES.resolve()
    assert output == OUTPUT.resolve()
    assert project == AI_TEST.resolve()


def test_geojson_inspection_accepts_polygons_and_clips_to_image(tmp_path: Path) -> None:
    path = tmp_path / "bounds.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"label": "sample 1"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-5, -5], [80, 0], [80, 60], [-5, -5]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _inspect_geojson(path, 50, 40) == [
        {"label": "sample 1", "safe_label": "sample_1", "bbox": [0, 0, 50, 40]}
    ]


def test_geojson_rejects_duplicate_output_labels(tmp_path: Path) -> None:
    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 0]]]}
    path = tmp_path / "duplicates.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"label": "ROI 1"}, "geometry": polygon},
                    {"type": "Feature", "properties": {"label": "ROI_1"}, "geometry": polygon},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ROIActionError, match="same output name"):
        _inspect_geojson(path, 100, 100)
