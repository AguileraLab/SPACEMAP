"""Merge joins two touching shapes (two pieces of one cell); Split cuts a shape along a dragged line."""

import pytest
from shapely.geometry import Polygon

from spacemap.viewer import shapes


def square(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]


def area(coordinates):
    return Polygon(coordinates[0], coordinates[1:]).area


def test_merge_joins_touching_pieces_and_closes_a_small_gap():
    assert area(shapes.join(square(0, 0, 10, 10), square(10, 0, 20, 10))) == pytest.approx(200)
    # Outlines traced from neighbouring masks leave about a pixel between them; the gap is filled.
    joined = shapes.join(square(0, 0, 10, 10), square(11, 0, 20, 10))
    assert len(joined) == 1 and area(joined) > 195
    with pytest.raises(ValueError, match="apart"):
        shapes.join(square(0, 0, 10, 10), square(30, 0, 40, 10))


def test_split_cuts_right_across_even_when_the_line_stops_short():
    pieces = shapes.cut(square(0, 0, 20, 10), [[10, 2], [10, 8]])
    assert [round(area(piece)) for piece in pieces] == [100, 100]
    # In a U shape a cut through one arm leaves the other arm whole.
    u = [[[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30], [0, 0]]]
    assert [round(area(piece)) for piece in shapes.cut(u, [[-2, 20], [12, 20]])] == [600, 100]
    with pytest.raises(ValueError):
        shapes.cut(square(0, 0, 20, 10), [[30, 30], [40, 40]])
    with pytest.raises(ValueError):
        shapes.cut(square(0, 0, 20, 10), [[10, 5], [10.5, 5]])


def test_the_route_checks_its_input():
    from starlette.testclient import TestClient

    from spacemap_gui.viewer import api

    client = TestClient(api.app)
    merged = client.post("/shapes/edit", json={"op": "merge", "shapes": [square(0, 0, 10, 10), square(10, 0, 20, 10)]})
    assert merged.status_code == 200 and len(merged.json()["polygons"]) == 1
    cut = client.post("/shapes/edit", json={"op": "split", "shape": square(0, 0, 20, 10), "line": [[10, -5], [10, 15]]})
    assert cut.status_code == 200 and len(cut.json()["polygons"]) == 2
    for body in (
        {"op": "merge", "shapes": [square(0, 0, 10, 10)]},
        {"op": "split", "shape": [[[0, 0], [1, "x"], [2, 2]]], "line": [[0, 0], [5, 5]]},
        {"op": "split", "shape": square(0, 0, 20, 10), "line": [[10, -5]]},
        {"op": "delete"},
        [1, 2],
    ):
        answer = client.post("/shapes/edit", json=body)
        assert answer.status_code == 400 and answer.json()["error"]
