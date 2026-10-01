"""The same marker panel must be available in a clone and an installed wheel."""

import json

from spacemap import marker_config


def test_installed_defaults_do_not_require_the_source_checkout(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "spacemap"
    package.mkdir(parents=True)
    manifest = package / "marker_configs.json"
    manifest.write_text('{"profiles": {"default_phenotyping": {"corr_markers": ["CD3"]}}}')
    monkeypatch.setattr(marker_config, "__file__", str(package / "marker_config.py"))
    assert marker_config.marker_config_path() == manifest
    assert json.loads(manifest.read_text())["profiles"]["default_phenotyping"]["corr_markers"] == ["CD3"]


def test_editable_defaults_use_the_shared_root_manifest(tmp_path, monkeypatch):
    package = tmp_path / "src" / "spacemap"
    package.mkdir(parents=True)
    manifest = tmp_path / "marker_configs.json"
    manifest.write_text('{}')
    monkeypatch.setattr(marker_config, "__file__", str(package / "marker_config.py"))
    assert marker_config.marker_config_path() == manifest
