"""Marker views match RESOLVE, using only the current ROI and no label writes."""

import itertools
import time

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap.resolve_live import _normalise
from spacemap.viewer import marker_transforms as m


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    values = np.array([[1.0, 3.0, 4.0], [2.0, 7.0, 4.0], [9.0, 1.0, 4.0], [3.0, 5.0, 4.0]])
    frame = pd.DataFrame(np.vstack([values, [[9000.0, 9000.0, 9000.0]]]), columns=["A", "B", "C"])
    frame["batch_id"] = ["roi"] * 4 + ["other"]
    frame["label"] = [1, 2, 3, 4, 1]
    path = tmp_path / "cells.csv"
    frame.to_csv(path, index=False)
    dataio.ensure_fast_copy(path, force=True)
    settings = {
        "table": str(path),
        "roi": "roi",
        "markers": ["A", "B", "C"],
        "roi_column": "batch_id",
        "label_column": "label",
    }
    profile = {
        "status": "ready",
        "source": "table",
        "label": 3,
        "markers": [{"marker": name, "mean": float(v)} for name, v in zip(settings["markers"], values[2])],
    }
    return settings, profile, values


CHAINS = [list(itertools.compress(m.ORDER, mask)) for mask in itertools.product((False, True), repeat=5)]


@pytest.mark.parametrize("chain", CHAINS)
def test_all_chains_match_resolve_and_keep_raw_values(setup, chain):
    settings, profile, values = setup
    stats = m.statistics(settings)
    expected = np.arcsinh(values / 5) if "Arcsinh" in chain else values
    expected = _normalise(expected, [name for name in chain if name != "Arcsinh"])
    actual = m.transform(values[2], chain, stats)
    np.testing.assert_allclose(actual, expected[2], rtol=1e-12, atol=1e-12)
    assert profile["markers"][0]["mean"] == 9


def test_controls_sort_steps_and_raw_is_exclusive():
    assert m.steps(["CDF", "Arcsinh", "Z2", "Z1"]) == ["Arcsinh", "Z1", "Z2", "CDF"]
    assert m.steps(["Raw"]) == []
    for bad in (["Raw", "Z1"], ["Z1", "Z1"], ["Other"], "Z1"):
        with pytest.raises(ValueError):
            m.steps(bad)


def wait(profile, settings, chain):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = m.apply_profile(profile, chain, settings)
        if result["status"] != "preparing":
            return result
        time.sleep(0.01)
    raise AssertionError("Marker statistics did not finish")


def test_stats_cached_across_clicks_and_unrelated_labels(setup, monkeypatch):
    settings, profile, _ = setup
    first = wait(profile, settings, ["Z1"])
    assert first["status"] == "ready"

    def no_repeat(*args):
        raise AssertionError("Clicking another cell must reuse prepared statistics")

    monkeypatch.setattr(m, "statistics", no_repeat)
    labels.append(settings["table"], [{"row": 1, "column": "phenotype_ref1", "value": "New label"}])
    again = wait(profile, settings, ["Arcsinh", "Z1", "Z2", "CDF", "NLog"])
    assert again["status"] == "ready"
    assert again["steps"] == list(m.ORDER)
    assert "this ROI" in again["view_note"]


def test_pending_marker_edits_and_missing_values(setup):
    settings, profile, values = setup
    before = m.revision(settings)
    labels.append(
        settings["table"], [{"row": 0, "column": "A", "value": "30"}, {"row": 1, "column": "B", "value": "nan"}]
    )
    assert m.revision(settings) != before
    means, scales, counts = m.statistics(settings)
    assert means[0, 0] == pytest.approx((30 + 2 + 9 + 3) / 4)
    assert means[0, 1] == pytest.approx((3 + 1 + 5) / 3)
    assert counts.tolist() == [4, 3, 4]


def test_image_profile_transforms_use_signal_and_require_table_for_z1(setup):
    _, profile, _ = setup
    image = {
        **profile,
        "source": "image",
        "markers": [{"marker": "A", "mean": 99, "signal": 5}, {"marker": "B", "mean": 40, "signal": 15}],
    }
    result = m.apply_profile(image, ["Arcsinh", "Z2"])
    np.testing.assert_allclose([row["view"] for row in result["markers"]], [-1, 1])
    assert result["markers"][0]["mean"] == 99 and result["markers"][0]["signal"] == 5
    with pytest.raises(ValueError, match="Saved measurements"):
        m.apply_profile(image, ["Z1"])


def test_marker_sets_saved_at_the_same_time_are_all_kept(tmp_path):
    """Two saves at once (Save pressed twice) shared a temporary file: one failed and sets were lost."""
    import threading

    from spacemap.viewer import display

    path = tmp_path / "display" / "marker_sets.json"
    errors = []

    def save(index):
        try:
            display.save_set(path, f"set{index}", [{"name": "CD3", "color": "#ff0000"}])
        except Exception as error:  # noqa: BLE001 - the test counts every failure
            errors.append(error)

    threads = [threading.Thread(target=save, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(display.load_sets(path)) == 20
    assert not list(path.parent.glob(".*.tmp"))
