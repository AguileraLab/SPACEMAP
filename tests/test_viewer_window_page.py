"""The window's page asks for its scripts and styles with their file dates, and is never cached."""

import re

from starlette.testclient import TestClient

from spacemap_gui.viewer import api


def test_the_page_names_each_file_with_its_date():
    client = TestClient(api.app)
    page = client.get("/window/index.html")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-store"
    scripts = re.findall(r'src="(viewer/core\.js|studio\.js)\?v=(\d+)"', page.text)
    assert {name for name, _ in scripts} == {"viewer/core.js", "studio.js"} and all(int(stamp) > 0 for _, stamp in scripts)
    assert 'href="viewer.css?v=' in page.text and 'src="vendor/openseadragon.min.js?v=' in page.text
    script = client.get("/window/viewer/core.js")
    assert script.status_code == 200 and script.headers["cache-control"] == "no-cache"
