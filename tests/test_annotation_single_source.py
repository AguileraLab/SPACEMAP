"""Each step reads the one annotation file and writes it back.

The tab used to offer a choice of source per step, listing files that might not
exist and that in some cases were the previous step's labels rather than tissue.
One file carries through instead: step 1 creates it, and each later step rewrites
it, so what the next stage opens is never in doubt.
"""

import json

import pytest
import streamlit as st

from spacemap_gui.tabs.tab4_annotation import s2b_tissue_pieces as pieces
from spacemap_gui.tabs.tab4_annotation.common import geojson_path


def _polygon(x0, y0, side):
    ring = [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side], [x0, y0]]
    return {"type": "Polygon", "coordinates": [ring]}


def _collection(entries):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": geometry,
         "properties": {"classification": {"name": name}}}
        for geometry, name in entries
    ]}


@pytest.fixture
def output_dir(tmp_path):
    st.session_state["annot_ta_output_dir"] = str(tmp_path)
    yield tmp_path
    st.session_state.pop("annot_ta_output_dir", None)


def test_step_2b_offers_no_choice_of_source():
    assert not hasattr(pieces, "SOURCE_OPTIONS"), "the source picker is gone; one file carries through"
    assert pieces.SOURCE_SUFFIX == "valid_tissue"


def test_it_reads_the_annotation_when_it_has_not_run_before(output_dir):
    path = output_dir / "S1_valid_tissue.geojson"
    path.write_text(json.dumps(_collection([(_polygon(0, 0, 100), "Valid_Tissue")])))

    geometry, others, source = pieces._load_source_geometry("S1")

    assert source == str(path)
    assert others == {}
    assert geometry.area == pytest.approx(100 * 100)


def test_a_rerun_starts_from_the_tissue_the_first_run_was_given(output_dir):
    """Otherwise a loosened size threshold could never bring a piece back."""
    annotation = output_dir / "S1_valid_tissue.geojson"
    annotation.write_text(json.dumps(_collection([(_polygon(0, 0, 100), "Valid_Tissue")])))
    review = geojson_path("S1", pieces.REVIEW_SUFFIX)
    import os
    os.makedirs(os.path.dirname(review), exist_ok=True)
    with open(review, "w", encoding="utf-8") as handle:
        json.dump(_collection([
            (_polygon(0, 0, 100), pieces.KEPT_CLASS),
            (_polygon(200, 0, 50), pieces.REJECTED_CLASS),
            (_polygon(400, 0, 10), pieces.NECK_CLASS),
        ]), handle)

    geometry, others, source = pieces._load_source_geometry("S1")

    assert source == review
    # Kept plus rejected, and the neck marker left out of the tissue.
    assert geometry.area == pytest.approx(100 * 100 + 50 * 50)


def test_missing_annotation_says_which_step_to_run(output_dir):
    with pytest.raises(FileNotFoundError) as error:
        pieces._load_source_geometry("S1")

    assert "step 1 and step 2" in str(error.value)


def test_the_review_layer_is_not_the_annotation(output_dir):
    """The review layer holds rejects and necks, so it must stay out of the way."""
    annotation = geojson_path("S1", pieces.SOURCE_SUFFIX)
    review = geojson_path("S1", pieces.REVIEW_SUFFIX)

    assert annotation != review
    assert "intermediate" in review and "intermediate" not in annotation


def test_a_custom_file_offers_its_classes_and_carries_the_rest(tmp_path):
    """The whole point of loading a hand-drawn annotation: pick the container,
    and everything inside it follows the edits made to it."""
    source = tmp_path / "P1_merged.geojson"
    source.write_text(json.dumps(_collection([
        (_polygon(0, 0, 400), "tissue"),
        (_polygon(1000, 0, 100), "tissue"),
        (_polygon(50, 50, 100), "tumor"),
        (_polygon(1010, 10, 40), "tumor"),
    ])))

    classes = pieces.load_classes(str(source))
    assert sorted(classes) == ["tissue", "tumor"]

    geometry, others, path = pieces._load_source_geometry(
        "P1", {"path": str(source), "class_name": "tissue"})

    assert path == str(source)
    assert sorted(others) == ["tumor"]
    assert geometry.area == pytest.approx(400 * 400 + 100 * 100)


def test_each_class_is_written_back_as_one_annotation(tmp_path, output_dir):
    """A hand-drawn class is one annotation that happens to have many parts.

    Exploding it into a feature per part would change what the file is rather
    than where its edges fall, so the cut is applied and the class stays whole.
    """
    from shapely.geometry import shape

    source = output_dir / "P9_merged.geojson"
    source.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"classification": {"name": "tissue", "colorRGB": -16711936}},
         "geometry": {"type": "MultiPolygon", "coordinates": [
             _polygon(0, 0, 600)["coordinates"], _polygon(2000, 0, 120)["coordinates"]]}},
        {"type": "Feature", "properties": {"classification": {"name": "tumor", "colorRGB": None}},
         "geometry": {"type": "GeometryCollection", "geometries": [
             _polygon(10, 10, 100), _polygon(200, 200, 100),
             {"type": "Point", "coordinates": [5, 5]},
             _polygon(2010, 10, 40)]}},
    ]}))

    config = {"microns_per_pixel": 0.5083, "detect_necks": False, "cut_at_necks": False,
              "filter_by_area": True, "min_area_mm2": 0.05}
    result = pieces._run_single("P9", config, {
        "path": str(source), "class_name": "tissue", "apply_to_others": True})

    written = json.loads(open(result["annotation_path"]).read())
    by_class = {}
    for feature in written["features"]:
        name = feature["properties"]["classification"]["name"]
        assert name not in by_class, f"{name} was split into more than one annotation"
        by_class[name] = feature

    assert sorted(by_class) == ["tissue", "tumor"]
    for name, feature in by_class.items():
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon"), (
            f"{name} kept a stray point or line from the source"
        )
    # The colour the file carried is kept; the one it lacked gets a distinct new one.
    assert by_class["tissue"]["properties"]["classification"]["colorRGB"] == -16711936
    tumor_colour = by_class["tumor"]["properties"]["classification"]["colorRGB"]
    assert tumor_colour is not None and tumor_colour != -16711936

    # The small tissue piece was dropped, so the tumour inside it goes too.
    tumour = shape(by_class["tumor"]["geometry"])
    assert tumour.bounds[2] < 2000, "tumour inside a dropped piece survived"
    assert tumour.area == pytest.approx(2 * 100 * 100, rel=0.01)
