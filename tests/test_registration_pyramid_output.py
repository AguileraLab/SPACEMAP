"""Guards on what the registration writers put on disk.

The pipeline was changed to emit tiled, pyramidal TIFFs so that palom reads its inputs
as dask chunks with real thumbnail levels, and so that ROI extraction can crop without
decoding full image widths. These tests pin the properties that change had to preserve:
full-resolution pixels, channel names, colors, Fluor and pixel size.
"""

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
import tifffile

from spacemap.registration import (
    _iter_plane_tiles,
    _level_spatial_shape,
    _pack_cached_plane,
    _unpack_cached_plane,
    _write_ome_pyramid_from_plane_factory,
    _write_tiled_pyramid,
)
from spacemap.utils import rgb_to_color_int


OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

CHANNEL_NAMES = ["DAPI", "CD3e", "Pan-Cytokeratin", "HE_Red"]
CHANNEL_COLORS = [(0, 0, 255), (255, 16, 32), (8, 240, 128), (255, 0, 0)]
CHANNEL_METADATA = [
    {"dye": "DAPI", "exposure": "5"},
    {"dye": "Opal520", "exposure": "7"},
    {"dye": "AF647", "exposure": "3"},
    {"dye": "-", "exposure": "-"},
]
PIXEL_SIZE = 0.5078595280647278


def _ome_channels(path):
    with tifffile.TiffFile(path) as handle:
        root = ET.fromstring(handle.ome_metadata)
    return root.findall(".//ome:Channel", OME_NS)


def _plane_factory(planes):
    def factory(_level, level_shape):
        height, width = level_shape
        for plane in planes:
            if plane.shape == (height, width):
                yield plane
            else:
                yield cv2.resize(plane, (width, height), interpolation=cv2.INTER_AREA)

    return factory


@pytest.fixture
def planes():
    rng = np.random.default_rng(20260825)
    return [rng.integers(0, 65535, (900, 700), dtype=np.uint16) for _ in CHANNEL_NAMES]


@pytest.mark.parametrize("shape", [(900, 700), (300, 200)])
def test_ome_pyramid_preserves_pixels_and_metadata(tmp_path, shape):
    """Full-resolution pixels and every OME channel attribute survive the tiled write.

    The small shape covers levels smaller than one tile, where every tile is padded.
    """
    rng = np.random.default_rng(7)
    height, width = shape
    source = [rng.integers(0, 65535, shape, dtype=np.uint16) for _ in CHANNEL_NAMES]
    output = tmp_path / "merged.ome.tiff"

    _write_ome_pyramid_from_plane_factory(
        output_path=str(output),
        plane_factory=_plane_factory(source),
        channel_names=CHANNEL_NAMES,
        channel_colors=CHANNEL_COLORS,
        pixel_size=PIXEL_SIZE,
        base_shape=shape,
        channel_metadata=CHANNEL_METADATA,
        n_pyramid_levels=3,
    )

    with tifffile.TiffFile(output) as handle:
        series = handle.series[0]
        assert handle.pages[0].is_tiled
        assert [level.shape for level in series.levels] == [
            (len(CHANNEL_NAMES), height, width),
            (len(CHANNEL_NAMES), height // 2, width // 2),
            (len(CHANNEL_NAMES), height // 4, width // 4),
            (len(CHANNEL_NAMES), height // 8, width // 8),
        ]
        written = series.levels[0].asarray()
        root = ET.fromstring(handle.ome_metadata)

    for index, plane in enumerate(source):
        np.testing.assert_array_equal(written[index], plane)

    channels = root.findall(".//ome:Channel", OME_NS)
    assert [channel.get("Name") for channel in channels] == CHANNEL_NAMES
    assert [channel.get("Color") for channel in channels] == [
        str(rgb_to_color_int(color)) for color in CHANNEL_COLORS
    ]
    assert [channel.get("Fluor") for channel in channels] == ["DAPI", "Opal520", "AF647", "-"]

    pixels = root.find(".//ome:Pixels", OME_NS)
    assert float(pixels.get("PhysicalSizeX")) == pytest.approx(PIXEL_SIZE)
    assert float(pixels.get("PhysicalSizeY")) == pytest.approx(PIXEL_SIZE)
    assert pixels.get("PhysicalSizeXUnit") == "µm"


@pytest.mark.parametrize(
    "image_shape, dtype, photometric, axes",
    [
        ((900, 700), np.uint16, "minisblack", "YX"),
        ((900, 700, 3), np.uint8, "rgb", "YXC"),
    ],
)
def test_tiled_pyramid_intermediates_expose_levels(tmp_path, image_shape, dtype, photometric, axes):
    """palom needs stored levels; an RGB plane labelled 'YXC' would hide them."""
    rng = np.random.default_rng(11)
    image = rng.integers(0, np.iinfo(dtype).max, image_shape, dtype=dtype)
    output = tmp_path / "intermediate.tiff"

    _write_tiled_pyramid(str(output), image, photometric=photometric, axes=axes, n_pyramid_levels=4)

    with tifffile.TiffFile(output) as handle:
        series = handle.series[0]
        assert handle.pages[0].is_tiled
        assert len(series.levels) == 5, "pyramid levels must be readable as series.levels"
        np.testing.assert_array_equal(series.levels[0].asarray(), image)
        assert series.levels[2].shape[:2] == (image_shape[0] // 4, image_shape[1] // 4)


def test_plane_tiles_cover_plane_with_zero_padding():
    plane = np.arange(7 * 5, dtype=np.uint16).reshape(7, 5)
    tiles = list(_iter_plane_tiles(plane, (4, 4)))

    assert len(tiles) == 4  # 2 tile rows x 2 tile columns
    assert all(tile.shape == (4, 4, 1) for tile in tiles)
    np.testing.assert_array_equal(tiles[0][:4, :4, 0], plane[:4, :4])
    np.testing.assert_array_equal(tiles[3][3:, :, 0], 0)  # padded past the last row


def test_pyramid_cache_round_trips_exactly():
    rng = np.random.default_rng(3)
    plane = rng.integers(0, 4000, (400, 300), dtype=np.uint16)

    codec, payload, nbytes = _pack_cached_plane(plane)
    restored = _unpack_cached_plane(codec, payload, nbytes, plane.shape, plane.dtype)

    np.testing.assert_array_equal(restored, plane)


@pytest.mark.parametrize(
    "shape, axes, expected",
    [
        ((49, 53280, 21120), "CYX", (53280, 21120)),
        ((152640, 78720, 3), "YXS", (152640, 78720)),
        ((900, 700, 3), "", (900, 700)),
        ((900, 700), "", (900, 700)),
    ],
)
def test_level_spatial_shape_reads_metadata_only(shape, axes, expected):
    level = type("Level", (), {"shape": shape, "axes": axes})()

    assert _level_spatial_shape(level) == expected
