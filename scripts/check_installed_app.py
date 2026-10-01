#!/usr/bin/env python3
"""Open the installed application's welcome page without study data or AI."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="spacemap-startup-") as scratch:
        for name in ("SPACEMAP_CONFIG_DIR", "SPACEMAP_STATE_DIR", "SPACEMAP_CACHE_DIR",
                     "SPACEMAP_LOGS_DIR", "SPACEMAP_MODELS_DIR"):
            os.environ.pop(name, None)
        os.environ["SPACEMAP_HOME"] = str(Path(scratch) / "state")
        os.environ["SPACEMAP_ASSISTANT"] = "off"
        os.environ["SPACEMAP_PROJECTS_DIR"] = str(Path(scratch) / "experiments")
        (Path(scratch) / "experiments").mkdir()
        from spacemap_gui.launch import _apply_runtime_defaults, _apply_thread_caps, app_path

        _apply_thread_caps()
        _apply_runtime_defaults()
        from streamlit.testing.v1 import AppTest
        from spacemap import ai_settings
        from spacemap.marker_config import marker_config_path
        from spacemap_gui.tabs.tab7_phenotyping.heatmap_click_selector import heatmap_click_selector_available

        from importlib.metadata import version
        import spacemap
        from spacemap.doctor import OK, check_slide_reader
        assert spacemap.__version__ == version("spacemap-ai")
        slide_reader = check_slide_reader()
        assert slide_reader.status == OK, slide_reader.detail
        assert not ai_settings.is_enabled()
        assert marker_config_path().is_file()
        assert heatmap_click_selector_available(), "Compiled heatmap widget missing"
        app = AppTest.from_file(str(app_path()), default_timeout=180).run()
        assert not app.exception, [str(error) for error in app.exception]
        assert any(button.label == "Create and Initialize Project" for button in app.button)
        app.text_input(key="new_proj_name").input("InstallCheck")
        app.text_input(key="proj_parent_dir").input(str(Path(scratch) / "experiments"))
        next(button for button in app.button if button.label == "Create and Initialize Project").click().run()
        assert not app.exception, [str(error) for error in app.exception]
        assert not app.error, [str(error.value) for error in app.error]
        project = Path(scratch) / "experiments" / "project" / "InstallCheck"
        assert (project / "InstallCheck.json").is_file(), "New project configuration was not saved"
        assert (project / "outputs" / "registered").is_dir()
        import json
        state = Path(scratch) / "state" / "state"
        assert json.loads((state / "projects.json").read_text())["InstallCheck"] == str(project)
        assert json.loads((state / "active_session.json").read_text())["active_project_name"] == "InstallCheck"
        print("OK: installed app opens Project Workspace and creates a project with the assistant off")


if __name__ == "__main__":
    main()
