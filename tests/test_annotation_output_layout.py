"""The output folder holds the annotation; the layers explaining it live below it.

A five-step run writes a dozen GeoJSON layers per sample. Kept side by side they
bury the one file the next stage opens, which is what this layout exists to
prevent. The split is also load-bearing: `PRIMARY_GEOJSON_SUFFIXES` is exactly
the set of names segmentation and visualization search for, so anything moved
out of that set stops being found.
"""

import json
import os

import pytest

from spacemap_gui.tabs.tab4_annotation.common import (
    INTERMEDIATE_DIRNAME,
    INTERMEDIATE_GEOJSON_SUFFIXES,
    PRIMARY_GEOJSON_SUFFIXES,
    geojson_path,
    intermediate_file,
    tidy_output_directory,
)

# Kept in step with segmentation_cellpose.ANNOTATION_GEOJSON_SUFFIXES, which is
# how the next tab finds an annotation at all.
DOWNSTREAM_SUFFIXES = ("zones", "macro_domains", "valid_tissue_from_clusters_highres", "valid_tissue")


@pytest.mark.parametrize("suffix", sorted(PRIMARY_GEOJSON_SUFFIXES))
def test_primary_layers_stay_in_the_output_folder(tmp_path, suffix):
    path = geojson_path("S1", suffix, directory=str(tmp_path))
    assert os.path.dirname(path) == str(tmp_path)


@pytest.mark.parametrize("suffix", sorted(INTERMEDIATE_GEOJSON_SUFFIXES))
def test_review_layers_go_one_level_down(tmp_path, suffix):
    path = geojson_path("S1", suffix, directory=str(tmp_path))
    assert os.path.basename(os.path.dirname(path)) == INTERMEDIATE_DIRNAME


@pytest.mark.parametrize("suffix", DOWNSTREAM_SUFFIXES)
def test_every_name_the_next_tab_searches_for_is_primary(suffix):
    assert suffix in PRIMARY_GEOJSON_SUFFIXES, (
        f"segmentation looks for _{suffix}; moving it out of the output folder hides it"
    )


def test_an_older_project_keeps_opening_its_files_where_they_are(tmp_path):
    """Nothing has to be regenerated for a project written before the split."""
    legacy = tmp_path / "S1_detected_blur.geojson"
    legacy.write_text('{"type":"FeatureCollection","features":[]}')

    assert geojson_path("S1", "detected_blur", directory=str(tmp_path)) == str(legacy)


def test_the_new_location_wins_once_it_exists(tmp_path):
    legacy = tmp_path / "S1_detected_blur.geojson"
    legacy.write_text("{}")
    canonical = tmp_path / INTERMEDIATE_DIRNAME / "S1_detected_blur.geojson"
    canonical.parent.mkdir()
    canonical.write_text("{}")

    assert geojson_path("S1", "detected_blur", directory=str(tmp_path)) == str(canonical)


def test_tidying_moves_review_layers_and_leaves_the_annotation(tmp_path):
    for name in ("S1_valid_tissue.geojson", "S1_macro_domains.geojson", "S1_zones.geojson"):
        (tmp_path / name).write_text("{}")
    for name in ("S1_tissue_boundary.geojson", "S1_blood_clot.geojson",
                 "S1_detected_blur.geojson", "S1_raw_macro_clusters.geojson",
                 "S1_artifact_review.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "notes.txt").write_text("not ours")

    moved = tidy_output_directory(str(tmp_path))

    assert moved == 5
    remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert remaining == ["S1_macro_domains.geojson", "S1_valid_tissue.geojson",
                         "S1_zones.geojson", "notes.txt"]
    moved_names = sorted(p.name for p in (tmp_path / INTERMEDIATE_DIRNAME).iterdir())
    assert moved_names == ["S1_artifact_review.json", "S1_blood_clot.geojson",
                           "S1_detected_blur.geojson", "S1_raw_macro_clusters.geojson",
                           "S1_tissue_boundary.geojson"]


def test_tidying_drops_a_stray_when_the_real_one_is_already_below(tmp_path):
    (tmp_path / INTERMEDIATE_DIRNAME).mkdir()
    canonical = tmp_path / INTERMEDIATE_DIRNAME / "S1_detected_lint.geojson"
    canonical.write_text('{"canonical": true}')
    (tmp_path / "S1_detected_lint.geojson").write_text('{"stale": true}')

    tidy_output_directory(str(tmp_path))

    assert not (tmp_path / "S1_detected_lint.geojson").exists()
    assert json.loads(canonical.read_text()) == {"canonical": True}


def test_tidying_an_absent_folder_is_harmless(tmp_path):
    assert tidy_output_directory(str(tmp_path / "does_not_exist")) == 0


def test_side_files_follow_the_same_split(tmp_path):
    path = intermediate_file("S1", "artifact_review.json", directory=str(tmp_path))
    assert os.path.basename(os.path.dirname(path)) == INTERMEDIATE_DIRNAME
