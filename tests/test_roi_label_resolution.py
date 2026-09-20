"""ROI names must survive the trip out of QuPath.

QuPath writes an annotation's name to ``properties.name`` and its class to
``properties.classification.name``. The drawing canvas used to read only
``properties.label``, so every QuPath-authored ROI appeared as "ROI" in the tab while
the extraction path — which read ``name`` — used the real names. The two disagreed.
"""

import pytest

from spacemap.utils import extract_rois_with_geometry_from_geojson, resolve_roi_label
from spacemap_gui.tabs.tab3_roi_extraction import geojson_to_fabric_with_labels


def _feature(properties, x=0, y=0):
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10], [x, y]]],
        },
    }


@pytest.fixture
def qupath_collection():
    """Four annotations exported from QuPath, named and classified."""
    names = ["pre", "LN2", "LN1", "post"]
    return {
        "type": "FeatureCollection",
        "features": [
            _feature(
                {"objectType": "annotation", "name": name, "classification": {"name": name}},
                x=index * 20,
            )
            for index, name in enumerate(names)
        ],
    }


@pytest.mark.parametrize(
    "properties, expected",
    [
        ({"label": "post"}, "post"),                                     # generic exporters
        ({"name": "LN1"}, "LN1"),                                        # QuPath annotation name
        ({"classification": {"name": "Tumor"}}, "Tumor"),                # QuPath class only
        ({"name": "LN2", "classification": {"name": "Node"}}, "LN2"),    # name outranks class
        ({"name": "   ", "classification": {"name": "Stroma"}}, "Stroma"),  # blank name skipped
        ({"label": "", "name": "", "classification": None}, "ROI_4"),    # nothing usable
        ({}, "ROI_4"),
        (None, "ROI_4"),
    ],
)
def test_resolve_roi_label(properties, expected):
    assert resolve_roi_label(properties, 3) == expected


def test_canvas_and_extraction_agree_on_qupath_names(qupath_collection):
    """The bug was that these two disagreed; they must not drift apart again."""
    extraction_labels = [label for _bbox, label, _geom in
                         extract_rois_with_geometry_from_geojson(qupath_collection)]
    _drawing, canvas_labels = geojson_to_fabric_with_labels(qupath_collection)

    assert extraction_labels == ["pre", "LN2", "LN1", "post"]
    assert canvas_labels == extraction_labels


def test_unnamed_rois_get_stable_distinct_fallbacks():
    collection = {"type": "FeatureCollection",
                  "features": [_feature({}, x=i * 20) for i in range(3)]}

    labels = [label for _bbox, label, _geom in extract_rois_with_geometry_from_geojson(collection)]

    assert labels == ["ROI_1", "ROI_2", "ROI_3"]
