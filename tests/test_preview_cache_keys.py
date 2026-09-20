"""Cached preview helpers must actually key on the files they read.

Streamlit drops any ``cache_data`` / ``cache_resource`` argument whose name
starts with an underscore, because that is how unhashable arguments are passed.
A change detector named that way is therefore silently discarded, and the panel
keeps serving the first result it ever produced no matter how many times the
underlying GeoJSON is rewritten. The bug is invisible at the call site, which is
why it is worth a test rather than a comment.
"""

import ast
import pathlib

import pytest

TAB = pathlib.Path(__file__).resolve().parents[1] / "src/spacemap_gui/tabs/tab4_annotation"
CHANGE_DETECTOR_WORDS = ("mtime", "stamp", "size", "token", "version", "signature", "modified")


def _cached_functions(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            source = ast.dump(decorator)
            if "cache_data" in source or "cache_resource" in source:
                yield node
                break


@pytest.mark.parametrize("path", sorted(TAB.glob("*.py")), ids=lambda p: p.name)
def test_no_change_detector_is_excluded_from_its_cache_key(path):
    offenders = []
    for function in _cached_functions(path):
        arguments = function.args.args + function.args.kwonlyargs
        for argument in arguments:
            name = argument.arg
            if name.startswith("_") and any(word in name.lower() for word in CHANGE_DETECTOR_WORDS):
                offenders.append(f"{path.name}:{function.name}({name})")

    assert not offenders, (
        "these arguments look like change detectors but Streamlit will ignore them, "
        f"so the cache can never notice a rewritten file: {offenders}"
    )


def test_the_polygon_overlay_is_keyed_on_timestamp_and_size():
    """The overlay is what the preview panel shows after every run."""
    source = (TAB / "common.py").read_text()

    assert "os.path.getmtime(path)" in source and "os.path.getsize(path)" in source, (
        "the overlay cache key must include both, so a rerun within one clock tick "
        "still redraws"
    )
    assert "    stamps: tuple[float, ...]," in source
