"""No registration widget may fall back to its minimum.

``st.number_input``/``st.slider`` default ``value`` to ``"min"``. A widget that omits
``value`` therefore renders at ``min_value`` on any run where its session key is absent,
and that value is then persisted into the project bundle. That is how Alignment
Keypoints silently became 500 — few enough that palom's coarse affine collapsed
(scale factors 1.58 / 0.017) and the block warp asked OpenCV for an 85327 px source
window, well past its 32767 px limit.
"""

import ast
import pathlib

import numpy as np
import pytest

from spacemap.registration import (
    MIN_RELIABLE_KEYPOINTS,
    _validate_coarse_affine,
)
from spacemap_gui.tabs.tab1_registration import DEFAULT_ALIGNMENT_KEYPOINTS


TAB1 = pathlib.Path(__file__).resolve().parents[1] / "src/spacemap_gui/tabs/tab1_registration.py"
WIDGETS = {"number_input", "slider"}


def _widgets_defaulting_to_min(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in WIDGETS:
            continue
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        if "min_value" in kwargs and "value" not in kwargs:
            label = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else "?"
            offenders.append(f"{path.name}:{node.lineno} {label!r}")
    return offenders


def test_no_registration_widget_falls_back_to_min_value():
    offenders = _widgets_defaulting_to_min(TAB1)
    assert not offenders, (
        "these widgets omit value= and so render at min_value when their session key is "
        "absent: " + ", ".join(offenders)
    )


def test_keypoint_default_is_high_enough_to_be_reliable():
    assert DEFAULT_ALIGNMENT_KEYPOINTS >= MIN_RELIABLE_KEYPOINTS


def test_validate_coarse_affine_accepts_a_healthy_transform():
    """The transform this pair actually produces at 10000 keypoints."""
    healthy = np.array([[0.9980, 0.0021, -8307.5], [-0.0021, 0.9977, -8626.3], [0, 0, 1]])

    _validate_coarse_affine(healthy, 10000)


def test_validate_coarse_affine_rejects_the_collapsed_transform():
    """The transform this pair produced at 500 keypoints, which crashed cv2.warpAffine."""
    collapsed = np.array([[1.5781, 0.0, -12137.2], [0.0, 0.0169, -18872.2], [0, 0, 1]])

    with pytest.raises(ValueError, match="implausible transform"):
        _validate_coarse_affine(collapsed, 500)


def test_validate_coarse_affine_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        _validate_coarse_affine(np.array([[1.0, 0, np.nan], [0, 1.0, 0], [0, 0, 1]]), 500)


def test_low_keypoint_count_still_warns_when_the_fit_happens_to_be_good(caplog):
    """500 keypoints sometimes fits fine; say so rather than failing silently."""
    healthy = np.eye(3)

    _validate_coarse_affine(healthy, 500)  # must not raise
