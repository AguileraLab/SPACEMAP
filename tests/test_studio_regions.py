"""Studio's regions: marker regions found with levels, and what stands out inside a region."""

import json

import numpy as np
import pytest
import tifffile


def _image(path):
    """Three markers: PanCK in the left half, ECad in the top half; CD3 bright only in a small square."""
    planes = np.full((3, 400, 400), 20, np.uint16)
    planes[0, :, :200] = 500  # PanCK
    planes[1, :200, :] = 800  # ECad
    planes[2, 300:320, 300:320] = 900  # CD3, a 20 px square
    tifffile.imwrite(
        path,
        planes,
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 1.0,
            "PhysicalSizeY": 1.0,
            "Channel": {"Name": ["PanCK", "ECad", "CD3"]},
        },
    )
    return path


def test_marker_regions_need_every_level_and_stay_inside_the_tissue(tmp_path):
    from shapely.geometry import shape

    from spacemap_gui.studio import regions

    image = _image(tmp_path / "S1.ome.tiff")
    values = {
        "rules": [{"marker": "PanCK", "above": 100}, {"marker": "ECad", "above": 200}],
        "sigma": 0,
        "within": "image",
        "min_area_um2": 10,
        "scale": 1,
    }
    features, stats = regions.find(image, values)
    assert len(features) == 1 and stats["regions"] == 1  # only the top-left quarter holds both
    assert abs(shape(features[0]["geometry"]).area - 200 * 200) < 400
    assert features[0]["properties"]["name"] == "Region 1" and not features[0]["properties"].get("named")
    # Inside the tissue only: a tissue covering the left 100 px halves it.
    tissue = tmp_path / "tissue.geojson"
    tissue.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [100, 0], [100, 400], [0, 400], [0, 0]]],
                        },
                    }
                ],
            }
        )
    )
    features, _ = regions.find(image, {**values, "within": "tissue", "name": "Tumor"}, tissue)
    assert abs(shape(features[0]["geometry"]).area - 100 * 200) < 400 and features[0]["properties"]["named"]
    # The smallest size drops small regions; a missing tissue or an unknown marker is explained.
    small, _ = regions.find(image, {**values, "rules": [{"marker": "CD3", "above": 100}], "min_area_um2": 500})
    assert small == []
    with pytest.raises(ValueError):
        regions.find(image, {**values, "within": "tissue"}, tmp_path / "none.geojson")
    with pytest.raises(ValueError):
        regions.find(image, {**values, "rules": [{"marker": "Nope", "above": 1}]})
    with pytest.raises(ValueError):
        regions.check({**values, "scale": 3})


def test_a_regions_card_ranks_the_markers_high_in_it(tmp_path):
    from spacemap_gui.studio import regions

    image = _image(tmp_path / "S2.ome.tiff")
    square = {"type": "Polygon", "coordinates": [[[300, 300], [320, 300], [320, 320], [300, 320], [300, 300]]]}
    found = regions.profile(image, square)
    assert found["markers"][0]["name"] == "CD3" and found["markers"][0]["z"] > 3 and found["markers"][0]["fold"] > 5
    assert found["reference"] == "whole image"


def test_marker_region_settings_are_saved(tmp_path):
    from spacemap_gui.studio import regions
    from spacemap_gui.studio.settings import read_config

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(
        json.dumps({"project_name": "study", "directory_mappings": {}, "tab_configurations": {}})
    )
    regions.save(root, {"rules": [{"marker": "PanCK", "above": "150"}], "sigma": 1.5})
    values = regions.settings(read_config(root))
    assert values["rules"] == [{"marker": "PanCK", "above": 150.0}] and values["sigma"] == 1.5 and values["scale"] == 4


def test_imported_annotation_inspection_respects_holes_hidden_classes_and_file_changes(tmp_path):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import api, registry

    image = _image(tmp_path / 'inspect.ome.tiff')
    outer = {'type':'Feature', 'properties':{'classification':{'name':'tissue'}}, 'geometry':{
        'type':'Polygon','coordinates':[[[0,0],[399,0],[399,399],[0,399],[0,0]],
                                      [[100,100],[200,100],[200,200],[100,200],[100,100]]]}}
    spot = {'type':'Feature','properties':{'classification':{'name':'lint'}},'geometry':{
        'type':'MultiPolygon','coordinates':[[[[300,300],[320,300],[320,320],[300,320],[300,300]]]]}}
    path = tmp_path / 'external.geojson'
    original = json.dumps({'type':'FeatureCollection','features':[outer,spot]})
    path.write_text(original)
    token = registry.register_object('annotations', path=str(path), image=str(image))
    client = TestClient(api.app)
    base = '/annotations/' + token
    hit = client.get(base+'/at?x=310&y=310').json()
    assert hit['feature']['properties']['name'] == 'lint'
    assert hit['feature']['geometry']['type'] == 'MultiPolygon'
    profile = client.post(base+'/profile', json={'index':hit['index'], 'stamp':hit['stamp']})
    assert profile.status_code == 200 and profile.json()['markers'][0]['name'] == 'CD3'
    assert client.get(base+'/at?x=310&y=310&hide=lint').json()['feature']['properties']['name'] == 'tissue'
    assert client.get(base+'/at?x=150&y=150').json()['feature'] is None  # hole
    assert client.get(base+'/at?x=450&y=450').json()['feature'] is None
    assert client.get(base+'/at?x=nan&y=0').status_code == 400
    assert client.post(base+'/profile', json={'index':-1,'stamp':hit['stamp']}).status_code == 400
    assert client.get('/annotations/missing/at?x=0&y=0').status_code == 404
    assert path.read_text() == original  # Inspection does not import, rewrite or save the source.
    path.write_text(json.dumps({'type':'FeatureCollection','features':[spot]}))
    assert client.post(base+'/profile', json={'index':hit['index'],'stamp':hit['stamp']}).status_code == 400
    assert client.get(base+'/at?x=310&y=310').json()['index'] == 0
