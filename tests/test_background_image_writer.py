import xml.etree.ElementTree as ET

import numpy as np
import pytest
import tifffile

from spacemap.background_detection import subtract_background_image
from spacemap.tissue_annotation_utils import PyramidImageReader
from spacemap.utils import rgb_to_color_int


OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def _write_source_ome(path, data):
    channel_names = ["CD3e", "CD8"]
    channel_colors = [(255, 16, 32), (8, 128, 240)]
    metadata = {
        "axes": "CYX",
        "PhysicalSizeX": 0.42,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": 0.42,
        "PhysicalSizeYUnit": "µm",
        "Channel": {
            "Name": channel_names,
            "Color": [rgb_to_color_int(color) for color in channel_colors],
            "Fluor": ["Dye-CD3", "Dye-CD8"],
        },
    }
    tifffile.imwrite(
        path,
        data,
        ome=True,
        photometric="minisblack",
        metadata=metadata,
    )
    return channel_names, channel_colors


@pytest.fixture()
def background_subtracted_ome(tmp_path):
    height, width = 32, 48
    cd3e = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    cd8 = np.full((height, width), np.iinfo(np.uint16).max, dtype=np.uint16)
    source_data = np.stack([cd3e, cd8])
    source = tmp_path / "source.ome.tiff"
    output = tmp_path / "source_bg_subtracted.ome.tiff"
    names, colors = _write_source_ome(source, source_data)

    returned = subtract_background_image(
        source,
        {"CD3e": 10.5},
        output,
    )

    assert returned == output
    assert output.is_file()
    return output, source_data, names, colors


def test_background_subtracted_image_is_roi_style_pyramidal_ome_tiff(
    background_subtracted_ome,
):
    output, _source_data, names, _colors = background_subtracted_ome

    with tifffile.TiffFile(output) as tif:
        assert tif.is_ome
        assert tif.is_bigtiff
        # Match the ROI exporter: one full-resolution base IFD per channel,
        # with three 2x reduced SubIFDs attached to every base IFD.
        assert len(tif.pages) == len(names)
        expected_shapes = [(16, 24), (8, 12), (4, 6)]
        for page in tif.pages:
            assert page.shape == (32, 48)
            assert page.is_tiled
            assert page.compression.name == "DEFLATE"
            assert page.subifds is not None
            assert len(page.subifds) == 3
            assert [level.shape for level in page.pages] == expected_shapes

    reader = PyramidImageReader(output)
    assert reader.is_pyramidal
    assert reader.level_count == 4
    assert reader.level_shapes == [(32, 48), *expected_shapes]


def test_background_subtracted_image_preserves_ome_channel_metadata(
    background_subtracted_ome,
):
    output, _source_data, names, colors = background_subtracted_ome

    with tifffile.TiffFile(output) as tif:
        root = ET.fromstring(tif.ome_metadata)
        pixels = root.find(".//ome:Pixels", OME_NS)
        channels = root.findall(".//ome:Channel", OME_NS)

    assert pixels is not None
    assert pixels.attrib["SizeX"] == "48"
    assert pixels.attrib["SizeY"] == "32"
    assert pixels.attrib["SizeC"] == "2"
    assert pixels.attrib["Type"] == "uint16"
    assert float(pixels.attrib["PhysicalSizeX"]) == pytest.approx(0.42)
    assert float(pixels.attrib["PhysicalSizeY"]) == pytest.approx(0.42)
    assert [channel.attrib.get("Name") for channel in channels] == names
    assert [int(channel.attrib["Color"]) for channel in channels] == [
        rgb_to_color_int(color) for color in colors
    ]
    assert [channel.attrib.get("Fluor") for channel in channels] == [
        "Dye-CD3",
        "Dye-CD8",
    ]


def test_background_subtracted_image_preserves_pixels_dtype_and_dimensions(
    background_subtracted_ome,
):
    output, source_data, _names, _colors = background_subtracted_ome

    with tifffile.TiffFile(output) as tif:
        written = np.stack([page.asarray() for page in tif.pages])

    expected_cd3e = np.maximum(source_data[0].astype(np.float32) - 10.5, 0).astype(
        np.uint16
    )
    assert written.shape == source_data.shape
    assert written.dtype == source_data.dtype
    np.testing.assert_array_equal(written[0], expected_cd3e)
    # A channel with no configured threshold is copied losslessly, including
    # the maximum value of its storage type.
    np.testing.assert_array_equal(written[1], source_data[1])
    assert written[0, 0, 0] == 0
