"""Channel discovery for TIFFs that carry neither OME-XML nor QPTIFF metadata.

The registration reference written by ``extract_dapi_channel`` is a single plane with
only shaped metadata. Before this fallback covered 2D images, ``get_channel_info``
raised and the GUI logged "Error reading channel info" after every successful
registration — alarming, and it left callers with an empty channel list.
"""

import numpy as np
import pytest
import tifffile

from spacemap.background_subtraction_utils import (
    _build_generic_channel_records,
    get_channel_info,
)


@pytest.fixture
def plain_tiff(tmp_path):
    def _write(array, photometric, axes):
        path = tmp_path / f"plain_{axes}.tiff"
        tifffile.imwrite(path, array, photometric=photometric, metadata={"axes": axes})
        return path

    return _write


def test_single_plane_grayscale_reports_one_channel(plain_tiff):
    """The shape of the registration reference: one plane, no channel axis."""
    path = plain_tiff(np.zeros((64, 48), dtype=np.uint16), "minisblack", "YX")

    names, colors, metadata, _mpp = get_channel_info(str(path))

    assert names == ["Channel_0"]
    assert colors == [(255, 255, 255)]
    assert metadata[0]["metadata_source"] == "generic-tiff"


def test_rgb_still_resolves_to_he_channel_names(plain_tiff):
    """Unchanged: a 3-sample plane is still recognised as RGB, not as one channel."""
    path = plain_tiff(np.zeros((64, 48, 3), dtype=np.uint8), "rgb", "YXS")

    names, _colors, _metadata, _mpp = get_channel_info(str(path))

    assert names == ["HE_Red", "HE_Green", "HE_Blue"]


@pytest.mark.parametrize(
    "shape, axes, expected",
    [
        ((64, 48), "YX", 1),          # single plane
        ((64, 48), "", 1),            # single plane, axes unknown
        ((4, 64, 48), "CYX", 4),      # channel-first stack
        ((64, 48, 3), "YXS", 3),      # RGB
        ((64,), "X", 0),              # not an image; still refuses to guess
    ],
)
def test_generic_records_channel_counts(shape, axes, expected):
    assert len(_build_generic_channel_records(shape, axes)) == expected
