"""Large annotations are read once even when several tiles arrive together."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from spacemap.viewer import annotations


def write_annotation(path, name="tissue"):
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"classification": {"name": name}},
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 0]]]},
                    }
                ],
            }
        )
    )


def test_simultaneous_annotation_requests_share_one_read(tmp_path, monkeypatch):
    path = tmp_path / "tissue.geojson"
    write_annotation(path)
    entered, release = threading.Event(), threading.Event()
    original = annotations._read_annotations
    reads = []

    def read(*args):
        reads.append(args)
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(annotations, "_read_annotations", read)
    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(annotations.load_annotations, path)
        assert entered.wait(5)
        others = [pool.submit(annotations.load_annotations, path) for _ in range(7)]
        release.set()
        results = [first.result(), *[job.result() for job in others]]
    assert len(reads) == 1
    assert all(result is results[0] for result in results)
    assert results[0].legend()[0]["value"] == "tissue"
    assert not annotations._LOADING


def test_changed_annotation_reloads_through_a_path_alias(tmp_path):
    path = tmp_path / "tissue.geojson"
    alias = tmp_path / "linked.geojson"
    write_annotation(path)
    alias.symlink_to(path)
    first = annotations.load_annotations(path)
    assert annotations.load_annotations(alias) is first
    write_annotation(path, "changed tissue class")
    updated = annotations.load_annotations(alias)
    assert updated is not first
    assert updated.legend()[0]["value"] == "changed tissue class"


def test_failed_annotation_read_can_be_retried(tmp_path):
    path = tmp_path / "tissue.geojson"
    path.write_text("broken json")
    with pytest.raises(ValueError):
        annotations.load_annotations(path)
    assert not annotations._LOADING
    write_annotation(path)
    assert annotations.load_annotations(path).legend()[0]["count"] == 1
