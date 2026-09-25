"""Every SPACEMAP page still renders: a guard for changes elsewhere (the image viewer, persistence).

Each page (and each Tab 7 method) is run once on a small synthetic project, in a
separate process with a scratch HOME, and must raise no exception.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    out = tmp_path_factory.mktemp("pages") / "pages.json"
    result = subprocess.run([sys.executable, str(REPO / "tests" / "page_smoke.py"), str(REPO), str(out)],
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(out.read_text())


PAGES = ["tab0", "tab1", "tab2", "tab3", "tab4", "tab5", "tab6", "tab8", "tab9", "app"] + [
    f"tab7:{method}" for method in ("leiden", "som", "vortex", "resolve", "scimap", "tools", "mrml",
                                    "consensus_ml", "subphenotyping")]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_an_exception(rendered, page):
    assert page in rendered, f"{page} was not rendered"
    assert rendered[page]["exceptions"] == [], rendered[page]["exceptions"]
    assert rendered[page]["widgets"] > 0
