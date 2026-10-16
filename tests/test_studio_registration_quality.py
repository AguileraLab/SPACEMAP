"""Studio's alignment check: DAPI against the H&E's nuclei tells an aligned pair from a shifted one."""

import numpy as np

from spacemap_gui.studio import registration
from test_viewer_cell_profiles import marker_image

NAMES = ["DAPI", "HE_Red", "HE_Green", "HE_Blue"]


def _pair(path, shift):
    """Round nuclei on a pale tissue: DAPI bright where the H&E is purple, the H&E moved by *shift* pixels."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:512, 0:512]
    nuclei = np.zeros((512, 512), bool)
    for cy, cx in rng.integers(20, 492, (220, 2)):
        nuclei |= (yy - cy) ** 2 + (xx - cx) ** 2 < 49
    moved = np.roll(nuclei, shift, axis=(0, 1))
    dapi = np.where(nuclei, 3000, 100).astype(np.uint16)
    he = np.stack([np.where(moved, 90, 235), np.where(moved, 40, 190), np.where(moved, 140, 215)]).astype(np.uint16)
    marker_image(path, np.concatenate([dapi[None], he]), NAMES)


def test_an_aligned_pair_scores_well_and_a_shifted_one_does_not(tmp_path):
    _pair(tmp_path / "good_merged.ome.tiff", (0, 0))
    _pair(tmp_path / "off_merged.ome.tiff", (9, 12))
    good = registration.quality(tmp_path / "good_merged.ome.tiff", side=512)
    off = registration.quality(tmp_path / "off_merged.ome.tiff", side=512)
    assert good["verdict"] == "good" and good["correlation"] > 0.8 and good["shift_um"] < 1
    assert off["verdict"] == "check" and off["correlation"] < 0.3 and 6 < off["shift_um"] < 9  # 15 px of 0.5083 µm


def test_the_checkerboard_leaves_every_other_square_see_through(tmp_path):
    import io

    from PIL import Image

    from spacemap.viewer import render
    from spacemap.viewer.sources import image_info

    _pair(tmp_path / "good_merged.ome.tiff", (0, 0))
    info = image_info(str(tmp_path / "good_merged.ome.tiff"))
    jpeg = render.render_tile(str(tmp_path / "good_merged.ome.tiff"), info, 0, 0, 0, None)
    alpha = np.array(Image.open(io.BytesIO(render.checker_tile(jpeg, info, 0, 0, 0, 64))))[..., 3]
    assert alpha[10, 10] == 255 and alpha[10, 70] == 0 and alpha[70, 10] == 0 and alpha[70, 70] == 255
    coarse = np.array(Image.open(io.BytesIO(render.checker_tile(jpeg, info, 1, 0, 0, 64))))[..., 3]
    assert coarse[10, 10] == 255 and coarse[10, 40] == 0  # a coarser tile: the squares keep their size on the tissue
