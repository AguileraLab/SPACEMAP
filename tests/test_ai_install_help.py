"""The assistant's setup page says exactly what to install, and where."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from spacemap import ai_settings

ROOT = Path(__file__).resolve().parents[1]


def _extra(name: str) -> dict[str, str]:
    """pip name -> version spec for one optional-dependency group of pyproject.toml."""
    text = (ROOT / "pyproject.toml").read_text()
    block = re.search(rf"^{name} = \[(.*?)\]", text, re.S | re.M).group(1)
    wanted = {}
    for item in re.findall(r'"([^"]+)"', block):
        requirement = item.split(";")[0].strip()
        match = re.match(r"([A-Za-z0-9_.-]+)(.*)", requirement)
        wanted[match.group(1)] = match.group(2)
    return wanted


def test_the_package_list_matches_pyproject():
    listed = {name: spec for name, spec, _, _ in ai_settings.AI_PACKAGES if name != "torch"}
    assert listed == _extra("ai")
    assert {name: spec for name, spec, _, _ in ai_settings.GGUF_PACKAGES} == _extra("gguf")
    assert "huggingface_hub" in ai_settings.LOCAL_BACKEND_MODULES  # downloading a model needs it
    assert "bitsandbytes" not in ai_settings.LOCAL_BACKEND_MODULES  # optional


def test_the_command_installs_only_what_is_missing_into_this_python():
    command = ai_settings.install_command([("transformers", ">=4.45"), ("fastapi", ">=0.110")])
    assert command == f'"{sys.executable}" -m pip install "transformers>=4.45" "fastapi>=0.110"'
    assert "spacemap-ai" not in command


def test_the_report_tells_installed_from_missing():
    report = ai_settings.dependency_report((("pytest", "", "pytest", True), ("not-a-package", "", "not_a_package_xyz", True)))
    assert report[0]["installed"] and report[1]["installed"] is None


def test_the_setup_page_shows_the_command(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def page():
        from spacemap import ai_settings
        from spacemap_gui.components import assistant_setup

        ai_settings.dependency_report = lambda packages=ai_settings.AI_PACKAGES: [
            {"name": "torch", "wanted": ">=2.0.0", "required": True, "installed": "2.12.1"},
            {"name": "transformers", "wanted": ">=4.45", "required": True, "installed": None},
            {"name": "bitsandbytes", "wanted": ">=0.43", "required": False, "installed": None},
        ]
        assistant_setup._render_dependency_status()

    app = AppTest.from_function(page, default_timeout=300)
    app.run()
    assert not app.exception
    commands = [block.value for block in app.code]
    assert any('-m pip install "transformers>=4.45"' in c and "torch" not in c for c in commands)
    assert any("same environment" in m.value or "inside the environment" in m.value for m in app.markdown)
