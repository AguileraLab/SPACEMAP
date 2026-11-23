"""Multiple nuclear signals reach every execution path without changing measurements."""
import json

import numpy as np
import pytest

from spacemap.segmentation_inputs import cellpose_input, maximum_projection, resolve_nuclear_indices
from spacemap_gui.studio import segmentation, training
from test_studio_artifacts import marker_image


def test_projection_retains_complementary_nuclei_without_rescaling_or_mutation():
    first = np.array([[0, 10], [3, 0]], dtype=np.uint16)
    second = np.array([[40, 2], [0, 9]], dtype=np.uint16)
    before = first.copy(), second.copy()
    signal = cellpose_input(iter([first, second]))
    expected = np.array([[40, 10], [3, 9]], dtype=np.float32)
    np.testing.assert_array_equal(signal[..., 0], expected)
    np.testing.assert_array_equal(signal[..., 2], expected)
    assert not signal[..., 1].any()
    np.testing.assert_array_equal(first, before[0])
    np.testing.assert_array_equal(second, before[1])
    np.testing.assert_array_equal(maximum_projection([first]), first)
    with pytest.raises(ValueError, match='at least one'):
        maximum_projection([])
    with pytest.raises(ValueError, match='dimensions'):
        maximum_projection([first, np.zeros((3, 3))])


def test_names_resolve_independently_of_image_channel_order():
    assert resolve_nuclear_indices(['CD31', 'Histone', 'DAPI'], ['DAPI', 'Histone', 'DAPI']) == [2, 1]
    assert resolve_nuclear_indices(['DAPI_1', 'Histone'], 'DAPI') == [0]
    assert resolve_nuclear_indices(['DAPI', 'Histone'], 'Histone') == [1]
    for selected in ([], ['DAPI', 'Missing']):
        with pytest.raises(ValueError):
            resolve_nuclear_indices(['DAPI', 'Histone'], selected)


@pytest.fixture
def scene(tmp_path):
    planes = np.zeros((3, 64, 64), dtype=np.uint16)
    planes[0, 12:20, 12:20] = 20
    planes[1, 40:48, 40:48] = 80
    planes[2] = np.arange(64)[None, :] + 1
    path = tmp_path / 'roi.ome.tiff'
    names = ['DAPI', 'Histone', 'Border']
    marker_image(path, planes, names)
    return path, names, planes


@pytest.fixture
def captured_segmenter(monkeypatch):
    from scipy.ndimage import label
    from spacemap import segmentation_cellpose as engine
    captured = []
    class Capture:
        def __init__(self, config):
            self.timings = {}
        def segment(self, picture, **kwargs):
            captured.append(picture.copy())
            return label(picture[..., 2] > 0)[0], [], None
    monkeypatch.setattr(engine, 'CellposeSegmenter', Capture)
    monkeypatch.setattr(segmentation, '_segmenter', Capture)
    return engine, captured


@pytest.mark.parametrize('borders', [[], ['Border']])
def test_preview_whole_image_roi_and_training_use_the_same_signal(scene, captured_segmenter, tmp_path, borders):
    import tifffile
    path, names, planes = scene
    engine, captured = captured_segmenter
    settings = {**segmentation.DEFAULTS, 'tab4_last_nuclear_names': ['DAPI', 'Histone'],
                'tab4_last_membrane_names': borders, 'tab4_gpu': False}
    expected_nucleus = np.maximum(planes[0], planes[1])
    expected_border = planes[2] if borders else expected_nucleus
    preview = segmentation.preview(str(path), names, [0, 0, 64, 64], settings)
    assert preview['count'] == 2 and preview['nuclear'] == ['DAPI', 'Histone']
    for masked in (False, True):
        config = engine.SegmentationConfig()
        config.use_gpu = config.save_geojson = False
        if masked:
            roi = tmp_path / 'tissue.geojson'
            roi.write_text(json.dumps({'type': 'FeatureCollection', 'features': [{
                'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Polygon',
                'coordinates': [[[0, 0], [64, 0], [64, 64], [0, 64], [0, 0]]]}}]}))
            config.use_input_geojson = True
            config.input_geojson_path = str(roi)
        result = engine.segment_image_cellpose(str(path), str(tmp_path / str(masked)), config,
            markers=borders, dapi_channel=['DAPI', 'Histone'])
        measured = result['features_df'].sort_values('y_centroid')
        assert len(measured) == 2
        # Measurements remain the original separate marker channels.
        np.testing.assert_allclose(measured['DAPI'], [20, 0])
        np.testing.assert_allclose(measured['Histone'], [0, 80])
    saved = training.save_example(tmp_path, str(path), names, [0, 0, 64, 64],
        [[[12, 12], [20, 12], [20, 20], [12, 20]]], settings)
    folder, _ = training.folders(tmp_path)
    picture = tifffile.imread(folder / (saved['saved'] + '_img.tif'))
    captured.append(np.moveaxis(picture, 0, -1))
    entry = json.loads((folder / 'examples.json').read_text())[0]
    assert entry['nuclear'] == ['DAPI', 'Histone']
    assert len(captured) == 4
    for picture in captured:
        np.testing.assert_array_equal(picture[..., 2], expected_nucleus)
        np.testing.assert_array_equal(picture[..., 0], expected_border)
        assert not picture[..., 1].any()


def test_studio_batch_passes_all_selected_nuclear_names(scene, monkeypatch, tmp_path):
    from spacemap import segmentation_cellpose as engine
    path, _, _ = scene
    calls = []
    monkeypatch.setattr(engine, 'segment_image_cellpose', lambda **kw: calls.append(kw) or {})
    settings = {**segmentation.values(tmp_path, {}), 'tab4_last_nuclear_names': ['DAPI', 'Histone'],
                'tab4_last_membrane_names': ['Border']}
    segmentation.run([{'image': str(path), 'outputs': str(tmp_path / 'output')}], settings, lambda *a: None)
    assert calls[0]['dapi_channel'] == ['DAPI', 'Histone'] and calls[0]['markers'] == ['Border']


@pytest.mark.parametrize('padding', [4, 64])
def test_tissue_area_preserves_bright_background_input_and_clips_saved_cells(tmp_path, monkeypatch, padding):
    """Area selection must not replace H&E's bright background with black.

    Exercise a partial ROI, an excluded hole and a cell crossing the boundary;
    the earlier all-image ROI test could not detect altered normalization.
    """
    from scipy.ndimage import label
    from shapely.geometry import Polygon
    from spacemap import segmentation_cellpose as engine

    names = ['DAPI', 'HE', 'Border']
    planes = np.zeros((3, 96, 96), dtype=np.uint16)
    planes[1] = 60000
    planes[2] = np.arange(96)[None, :] + 100
    # Inside, outside, in the excluded hole, and across the right boundary.
    for y, x in [(25, 25), (25, 10), (45, 45), (60, 68)]:
        planes[0, y:y+8, x:x+8] = 2000
        planes[1, y:y+8, x:x+8] = 48000
    path = tmp_path / 'bright.ome.tiff'
    marker_image(path, planes, names)
    captured = []

    class DarkNuclei:
        def __init__(self, config):
            self.timings = {}
        def segment(self, picture, **kwargs):
            captured.append(picture.copy())
            # Brightfield nuclei are darker than the surrounding image.
            return label(picture[..., 2] < 55000)[0], [], None

    monkeypatch.setattr(engine, 'CellposeSegmenter', DarkNuclei)
    monkeypatch.setattr(segmentation, '_segmenter', DarkNuclei)
    settings = {**segmentation.DEFAULTS, 'tab4_last_nuclear_names': ['DAPI', 'HE'],
                'tab4_last_membrane_names': ['Border'], 'tab4_gpu': False}
    assert segmentation.preview(str(path), names, [0, 0, 96, 96], settings)['count'] == 4
    roi = tmp_path / 'tissue.geojson'
    roi.write_text(json.dumps({'type': 'FeatureCollection', 'features': [{
        'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Polygon', 'coordinates': [
            [[20, 20], [72, 20], [72, 80], [20, 80], [20, 20]],
            [[40, 40], [56, 40], [56, 56], [40, 56], [40, 40]],
        ]}}]}))
    config = engine.SegmentationConfig()
    config.use_gpu = False
    config.use_input_geojson = True
    config.input_geojson_path = str(roi)
    config.tissue_crop_padding = padding
    config.save_npy = config.save_geojson = True
    output = tmp_path / 'output'
    result = engine.segment_image_cellpose(str(path), str(output), config,
                                         markers=['Border'], dapi_channel=['DAPI', 'HE'])
    area = engine.get_tissue_mask_for_tile_geometry((96, 96), (0, 0), engine.load_tissue_geometry(str(roi)))
    y0, y1, x0, x1 = engine.get_mask_crop_bounds(area, padding)
    np.testing.assert_array_equal(captured[1], captured[0][y0:y1, x0:x1])
    measured = result['features_df'].sort_values('y_centroid')
    assert len(measured) == 2
    np.testing.assert_allclose(measured['HE'], 48000)
    np.testing.assert_allclose(measured['DAPI'], 2000)
    np.testing.assert_allclose(measured['Border'], [128.5, 170])
    masks = np.load(next((output / 'npy').glob('*_mask.npy')))
    assert not masks[area == 0].any()
    assert masks[26, 26] > 0 and masks[62, 70] > 0
    assert masks[26, 11] == masks[46, 46] == masks[62, 75] == 0
    exported = json.loads(next((output / 'geojson').glob('*.geojson')).read_text())
    assert len(exported['features']) == 2
    # Raster polygon edges can be one pixel beyond the final pixel's centre.
    for feature in exported['features']:
        polygon = Polygon(feature['geometry']['coordinates'][0])
        assert polygon.bounds[2] <= 73 and polygon.bounds[0] >= 20


@pytest.mark.parametrize('problem', ['missing', 'invalid', 'empty', 'outside'])
def test_an_unusable_selected_area_never_falls_back_to_whole_image(scene, captured_segmenter, tmp_path, problem):
    path, _, _ = scene
    engine, captured = captured_segmenter
    area = tmp_path / 'selected.geojson'
    if problem == 'invalid':
        area.write_text('not valid GeoJSON')
    elif problem in ('empty', 'outside'):
        features = [] if problem == 'empty' else [{
            'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Polygon',
            'coordinates': [[[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]]}}]
        area.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))
    config = engine.SegmentationConfig()
    config.use_gpu = False
    config.use_input_geojson = True
    config.input_geojson_path = str(area)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='segmentation (area|GeoJSON)'):
        engine.segment_image_cellpose(str(path), str(output), config, markers=['Border'], dapi_channel=['DAPI'])
    assert not captured
    assert not list(output.rglob('*_cell_features.csv'))
