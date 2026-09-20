"""Test-wide setup that keeps the suite self-contained.

The assistant resolves its data directories once, at import time, from
``SPACEMAP_HOME``.  Pointing that at a throwaway directory here does two
things: the suite never reads or writes a developer's real SPACEMAP state, and
it can create the small project scaffold that the action tests expect instead
of depending on research data that cannot be published.

This module must set the environment before anything imports ``spacemap``,
which is why the work happens at import time rather than in a fixture.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# A session-lifetime directory.  pytest's tmp_path cannot be used because the
# value is needed before any fixture runs.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="spacemap-tests-"))
os.environ["SPACEMAP_HOME"] = str(_TEST_HOME)
# Keep the optional assistant switched off so that importing the GUI never
# depends on AI extras being installed.
os.environ.setdefault("SPACEMAP_ASSISTANT", "off")

#: Files the action tests reference by name.  Only the name and the fact that
#: the path exists matter: the validators check the suffix, not the bytes.
_SCAFFOLD_FILES = (
    "AI_test/data/codex/P11_codex.qptiff",
    "AI_test/data/he/P11_he.qptiff",
    "AI_test/data/P11_roi_bounding_boxes_coords.geojson",
    "AI_test/outputs/AI_registered/P11_merged.ome.tiff",
    "AI_test/registered.ome.tiff",
    "AI_test/cells.geojson",
    "study/outputs/rois/sample.ome.tiff",
    "study/outputs/annotations/sample_detected_blur.geojson",
)

_SCAFFOLD_DIRS = (
    "AI_test/outputs/rois",
    "AI_test/outputs/AI_registered",
    "study/outputs/processed_annotations",
)


def _build_scaffold() -> None:
    projects = _TEST_HOME / "projects"
    for relative in _SCAFFOLD_DIRS:
        (projects / relative).mkdir(parents=True, exist_ok=True)
    for relative in _SCAFFOLD_FILES:
        target = projects / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.touch()


_build_scaffold()


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    """Remove the throwaway home once the run is over."""
    import shutil

    shutil.rmtree(_TEST_HOME, ignore_errors=True)
