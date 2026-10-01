"""Locate the shared marker defaults in either a wheel or an editable clone."""

from pathlib import Path


def marker_config_path() -> Path:
    """Return the read-only manifest; project choices are saved separately."""
    packaged = Path(__file__).resolve().with_name("marker_configs.json")
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "marker_configs.json"
