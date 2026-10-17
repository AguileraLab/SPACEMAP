"""Native tiles retain the source display pixels; overview tiles retain chroma detail."""
import io

import numpy as np
from PIL import Image, JpegImagePlugin
import tifffile

from spacemap.viewer import render, view_state
from spacemap.viewer.sources import ImageInfo, Level, open_source
from spacemap_gui.viewer import registry
from test_viewer import client, _write_channel_pages


def test_native_rgb_tile_matches_tiff_exactly_and_overview_stays_jpeg(client, tmp_path):
    y, x = np.indices((601, 731))
    pixels = np.stack([x % 256, (x % 2) * 255, (y % 2) * 255], axis=-1).astype('uint8')
    path = tmp_path / 'detail.tif'
    with tifffile.TiffWriter(path) as writer:
        writer.write(pixels, photometric='rgb', tile=(128, 128), subifds=1)
        writer.write(pixels[::2, ::2], photometric='rgb', tile=(128, 128), subfiletype=1)
    token = registry.register(path)
    source = open_source(path)
    for tx, ty in ((0, 0), (1, 1)):
        answer = client.get(f'/image/{token}/tile/0/{tx}/{ty}.png')
        assert answer.status_code == 200 and answer.headers['content-type'] == 'image/png'
        rendered = np.asarray(Image.open(io.BytesIO(answer.content)))
        np.testing.assert_array_equal(rendered, pixels[ty*512:(ty+1)*512, tx*512:(tx+1)*512])
    overview = client.get(f'/image/{token}/tile/1/0/0.jpg')
    picture = Image.open(io.BytesIO(overview.content))
    assert overview.headers['content-type'] == 'image/jpeg' and picture.size == (366, 301)
    assert JpegImagePlugin.get_sampling(picture) == 0  # 4:4:4: do not blur the colour channels
    # PNG is also available at reduced resolutions for figure export.
    reduced = client.get(f'/image/{token}/tile/1/0/0.png')
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(reduced.content))),
                                  render.raw_tile(str(path), source.info, 1, 0, 0, 0))


def test_marker_composite_lossless_tiles_and_checkerboard(client, tmp_path):
    path = tmp_path / 'markers.ome.tiff'
    _write_channel_pages(path, channels=3, height=120, width=160)
    token = registry.register(path); info = open_source(path).info
    styles = [render.ChannelStyle(i, color, 0, 6000) for i, color in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1)))]
    expected = render.composite([render.raw_tile(str(path), info, 0, 0, 0, s.index) for s in styles], styles)
    url = f'/image/{token}/tile/0/0/0.png?s=0:ff0000:0:6000;1:00ff00:0:6000;2:0000ff:0:6000'
    result = client.get(url)
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(result.content))), expected)
    checker = np.asarray(Image.open(io.BytesIO(client.get(url + '&ck=32').content)))
    np.testing.assert_array_equal(checker[..., :3], expected)
    assert set(np.unique(checker[..., 3])) == {0, 255}
    hidden = client.get(f'/image/{token}/tile/0/0/0.png?s=')
    assert not np.asarray(Image.open(io.BytesIO(hidden.content))).any()


def test_pyramid_choice_does_not_enlarge_a_coarser_vertical_level():
    info = ImageInfo('', 100, 100, [Level(100, 100, 1, 1), Level(50, 25, 2, 4)], ['R'], 'uint8', False, None, '', '')
    assert render._pick_level(info, 2) == 0
    assert render._pick_level(info, 4) == 1


def test_smoothing_preference_is_saved_per_image(tmp_path):
    for enabled in (True, False):
        saved = view_state.save(tmp_path, tmp_path/'image.tif', {'pixelSmoothing': enabled})
        assert view_state.load(view_state.path_for(tmp_path, tmp_path/'image.tif'))['pixelSmoothing'] is enabled
    assert 'pixelSmoothing' not in view_state.clean({'pixelSmoothing': 'false'})
