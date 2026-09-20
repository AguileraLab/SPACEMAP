"""ROI selection and the boundary-source control.

Selecting an ROI is now the table's job, so the old six-way "boundary method" radio
collapsed to three sources plus an explicit draw toggle. Old projects still hold the old
strings — including "Select / Edit / Move existing boundaries", which was a canvas mode
rather than a source — and must map onto the new control without losing anything.
"""

import pytest

from spacemap_gui.tabs.tab3_roi_extraction import (
    BOUNDARY_METHOD_FILE,
    BOUNDARY_METHOD_LEGACY_TRANSFORM,
    BOUNDARY_METHOD_TMA,
    BOUNDARY_SOURCE_DRAW,
    BOUNDARY_SOURCE_FILE,
    BOUNDARY_SOURCE_TMA,
    DRAW_SHAPE_CANVAS_MODES,
    DRAW_SHAPE_METHODS,
    _roi_extent,
    boundary_shape_from_stored,
    boundary_source_from_stored,
)


@pytest.mark.parametrize(
    "stored, expected",
    [
        (BOUNDARY_METHOD_FILE, BOUNDARY_SOURCE_FILE),
        (BOUNDARY_METHOD_TMA, BOUNDARY_SOURCE_TMA),
        (DRAW_SHAPE_METHODS["Polygon"], BOUNDARY_SOURCE_DRAW),
        (DRAW_SHAPE_METHODS["Freehand"], BOUNDARY_SOURCE_DRAW),
        (DRAW_SHAPE_METHODS["Rectangle"], BOUNDARY_SOURCE_DRAW),
        (BOUNDARY_METHOD_LEGACY_TRANSFORM, BOUNDARY_SOURCE_DRAW),  # old projects
        (None, BOUNDARY_SOURCE_DRAW),
        ("something a future version wrote", BOUNDARY_SOURCE_DRAW),
    ],
)
def test_boundary_source_from_stored(stored, expected):
    assert boundary_source_from_stored(stored) == expected


@pytest.mark.parametrize(
    "stored, expected",
    [
        (DRAW_SHAPE_METHODS["Polygon"], "Polygon"),
        (DRAW_SHAPE_METHODS["Freehand"], "Freehand"),
        (DRAW_SHAPE_METHODS["Rectangle"], "Rectangle"),
        (BOUNDARY_METHOD_FILE, "Polygon"),                  # falls back to the default
        (BOUNDARY_METHOD_LEGACY_TRANSFORM, "Polygon"),
        (None, "Polygon"),
    ],
)
def test_boundary_shape_from_stored(stored, expected):
    assert boundary_shape_from_stored(stored) == expected


def test_every_shape_has_a_canvas_mode():
    assert set(DRAW_SHAPE_METHODS) == set(DRAW_SHAPE_CANVAS_MODES)


def test_stored_methods_round_trip_through_the_new_control():
    """Whatever a project stored must map to a source and back to the same string."""
    for shape, method in DRAW_SHAPE_METHODS.items():
        assert boundary_source_from_stored(method) == BOUNDARY_SOURCE_DRAW
        assert DRAW_SHAPE_METHODS[boundary_shape_from_stored(method)] == method


@pytest.mark.parametrize(
    "obj, expected",
    [
        ({"type": "rect", "width": 100, "height": 50}, (200, 100)),
        ({"type": "rect", "width": 100, "height": 50, "scaleX": 2.0, "scaleY": 0.5}, (400, 50)),
        ({"type": "path", "path": [["M", 0, 0], ["L", 30, 0], ["L", 30, 40], ["z"]]}, (60, 80)),
        ({"type": "polygon", "points": [{"x": 0, "y": 0}, {"x": 10, "y": 25}]}, (20, 50)),
    ],
)
def test_roi_extent_scales_to_image_pixels(obj, expected):
    assert _roi_extent(obj, scale_x=2.0, scale_y=2.0) == expected


@pytest.mark.parametrize(
    "obj",
    [
        {"type": "image"},                 # the background, never an ROI
        {"type": "path", "path": []},      # nothing to measure
        {},                                # malformed
    ],
)
def test_roi_extent_degrades_instead_of_raising(obj):
    """A shape the table cannot measure must not stop the table from rendering."""
    assert _roi_extent(obj) == (None, None)


def test_geometry_signature_ignores_styling_the_tab_rewrites():
    """Re-colouring an outline must not read as a geometry change."""
    from spacemap_gui.tabs.tab3_roi_extraction import _canvas_geometry_signature

    shape = {"type": "rect", "left": 10, "top": 20, "width": 100, "height": 50}
    styled = dict(shape, stroke="#ff0000", strokeWidth=4, fill="rgba(0,0,0,0.1)")

    assert _canvas_geometry_signature([shape]) == _canvas_geometry_signature([styled])


def test_geometry_signature_notices_a_resize():
    from spacemap_gui.tabs.tab3_roi_extraction import _canvas_geometry_signature

    before = [{"type": "rect", "left": 10, "top": 20, "width": 100, "height": 50}]
    after = [{"type": "rect", "left": 10, "top": 20, "width": 140, "height": 50}]

    assert _canvas_geometry_signature(before) != _canvas_geometry_signature(after)


def test_geometry_signature_skips_the_background_image():
    """The background is an object on the canvas but is never an ROI."""
    from spacemap_gui.tabs.tab3_roi_extraction import _canvas_geometry_signature

    shape = {"type": "rect", "left": 0, "top": 0, "width": 5, "height": 5}

    assert _canvas_geometry_signature([{"type": "image", "src": "x"}, shape]) == \
           _canvas_geometry_signature([shape])
