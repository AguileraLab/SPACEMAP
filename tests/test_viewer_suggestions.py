"""Suggestions use only chosen examples and never change labels themselves."""

import json
import time

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap.viewer import cells, reference, suggestions as s


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    from test_viewer_cell_profiles import marker_image, polygon, square

    image = tmp_path / "roi.ome.tiff"
    pixels = np.full((2, 64, 64), 60000, np.uint16)
    features = []
    values = [[100, 1], [90, 2], [1, 100], [2, 90], [95, 1], [30, 30]]
    for i, value in enumerate(values):
        x = i * 9 + 1
        features.append(polygon(i + 1, square(x, 2, 6)))
        pixels[:, 2:9, x : x + 7] = np.asarray(value)[:, None, None]
    marker_image(image, pixels)
    seg = tmp_path / "cells.geojson"
    seg.write_text(json.dumps({"features": features}))
    pyramid = cells.ensure_label_pyramid(seg, 64, 64, tmp_path / "cache")
    s.cell_profiles.build_index(seg, s.cell_profiles.index_path(seg))
    table = tmp_path / "cells.csv"
    pd.DataFrame(
        {
            "batch_id": ["roi"] * 6 + ["other"],
            "label": [1, 2, 3, 4, 5, 6, 5],
            "M0": [v[0] for v in values] + [999],
            "M1": [v[1] for v in values] + [999],
            "phenotype_ref1": ["T", "T", "B", "B", None, None, "OTHER"],
            "alternate": ["X", "X", "Y", "Y", None, None, "OTHER"],
        }
    ).to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    layer = {"image": str(image), "pyramid": str(pyramid), "segmentation": str(seg), "roi": "roi"}
    settings = s.settings_for(
        layer,
        str(table),
        {"source": "table", "column": "phenotype_ref1", "destination": "phenotype_ref1"},
        ["M0", "M1"],
    )
    return settings, layer


def test_probabilities_and_roi_identity(setup):
    settings, _ = setup
    result = s.run(settings, label=5)
    assert result["choices"][0]["phenotype"] == "T"
    assert sum(c["probability"] for c in result["choices"]) == pytest.approx(1)
    assert all(0 < c["probability"] < 1 for c in result["choices"])
    assert result["provenance"]["examples"] == 4
    assert not reference.path_for(settings["table"]).exists()
    assert not labels.sidecar_path(settings["table"]).exists()


def test_selected_column_and_latest_saved_labels_are_used(setup):
    settings, _ = setup
    assert s.run({**settings, "column": "alternate"}, label=5)["choices"][0]["phenotype"] == "X"
    previous = s.run(settings, label=5)
    reference.save(settings["table"], "roi", "phenotype_ref1", {1: "B", 2: "B", 3: "T", 4: "T"}, "swapped-labels")
    current = s.run(settings, label=5)
    assert current["choices"][0]["phenotype"] == "B"
    assert current["provenance"]["revision"] != previous["provenance"]["revision"]


def test_flagged_examples_are_not_used(setup):
    settings, _ = setup
    s.run(settings, label=5)
    reference.set_flag(settings["table"], "roi", "phenotype_ref1", 1, True)
    with pytest.raises(ValueError, match="two checked examples"):
        s.run(settings, label=5)
    reference.set_flag(settings["table"], "roi", "phenotype_ref1", 1, False)
    assert s.run(settings, label=5)["provenance"]["examples"] == 4


def test_a_rare_class_is_disclosed_without_blocking_supported_classes(setup):
    settings, _ = setup
    reference.save(settings["table"], "roi", "phenotype_ref1", {6: "Rare"}, "rare-example")
    result = s.run(settings, label=5)
    assert result["provenance"]["needs_examples"] == ["Rare"]
    assert {c["phenotype"] for c in result["choices"]} == {"T", "B"}


def test_a_cell_never_votes_for_itself():
    model = s.Model(
        np.arange(1, 5), np.asarray([[100, 0], [0, 100], [100, 0], [0, 100]]), np.asarray(["A", "A", "B", "B"])
    )
    own = model.predict([[100, 0]], [1])[0]
    other = model.predict([[100, 0]], [9])[0]
    assert own[0] < other[0]  # own A vote is excluded, even though it matches exactly


def test_least_sure_uses_only_unlabelled_cells_and_updates_after_save(setup):
    settings, _ = setup
    result = s.run(settings, scan=True)
    assert result["labels"].tolist() == [6, 5]
    reference.save(settings["table"], "roi", "phenotype_ref1", {6: "B"}, "new-label-six")
    assert s.run(settings, scan=True)["labels"].tolist() == [5]


def test_reference_edits_reuse_features_but_marker_edits_refresh_them(setup, monkeypatch):
    settings, _ = setup
    s.run(settings, scan=True)
    reference.save(settings["table"], "roi", "phenotype_ref1", {6: "B"}, "new-training-example")
    original = cells.fast_rows

    def no_marker_reread(table, positions, columns):
        assert not set(columns) & {"M0", "M1"}, "A label edit reread unchanged marker data"
        return original(table, positions, columns)

    monkeypatch.setattr(cells, "fast_rows", no_marker_reread)
    assert s.run(settings, scan=True)["labels"].tolist() == [5]
    monkeypatch.setattr(cells, "fast_rows", original)
    labels.append(
        settings["table"], [{"row": 4, "column": "M0", "value": "0"}, {"row": 4, "column": "M1", "value": "200"}]
    )
    assert s.run(settings, label=5)["choices"][0]["phenotype"] == "B"


def test_large_training_columns_disclose_repeatable_sampling(setup):
    settings, _ = setup
    table = settings["table"]
    pd.DataFrame(
        {
            "batch_id": ["roi"] * 402,
            "label": np.arange(1, 403),
            "M0": np.arange(402),
            "M1": np.arange(402),
            "phenotype_ref1": ["T"] * 200 + ["B"] * 200 + [None] * 2,
        }
    ).to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    first = s.run(settings, label=401)
    assert first["provenance"]["examples"] == 200
    assert first["provenance"]["available_examples"] == 400
    assert first == s.run(settings, label=401)


def test_missing_values_are_not_imputed_as_positive_or_zero():
    model = s.Model(np.arange(4), np.asarray([[1, 2], [2, 1], [9, 8], [8, 9]]), np.asarray(["A", "A", "B", "B"]))
    assert np.isnan(model.predict([[np.nan, 10]], [99])).all()


def test_image_source_measures_pixels_not_table_values(setup, monkeypatch):
    settings, layer = setup
    pixels = s.settings_for(
        layer,
        settings["table"],
        {
            "source": "image",
            "column": "phenotype_ref1",
            "destination": "phenotype_ref1",
            "backgrounds": {"M0": 0, "M1": 0},
        },
        [],
    )

    def no_table(*args):
        raise AssertionError("Image suggestions must not substitute table measurements")

    monkeypatch.setattr(s, "_table_values", no_table)
    assert s.run(pixels, label=5)["choices"][0]["phenotype"] == "T"
    assert s.run(pixels, scan=True)["labels"].tolist() == [6, 5]


def test_full_pixel_pass_matches_exact_clicked_cell_profiles(setup):
    settings, layer = setup
    pixels = s.settings_for(
        layer,
        settings["table"],
        {
            "source": "image",
            "column": "phenotype_ref1",
            "destination": "phenotype_ref1",
            "backgrounds": {"M0": 10.5, "M1": 3.25},
        },
        [],
    )
    ids, values = s._all_pixels(pixels, lambda text: None)
    for label, row in zip(ids, values):
        profile = s.cell_profiles.image_profile(
            layer["image"], layer["pyramid"], layer["segmentation"], label, pixels["backgrounds"]
        )
        np.testing.assert_allclose(row, [m["signal"] for m in profile["markers"]], rtol=0, atol=1e-12)
    assert s._all_pixels(pixels, lambda text: None)[1] is values


def test_image_source_requires_all_backgrounds(setup):
    settings, layer = setup
    with pytest.raises(ValueError, match="every marker"):
        s.settings_for(
            layer,
            settings["table"],
            {"source": "image", "column": "phenotype_ref1", "destination": "phenotype_ref1"},
            [],
        )


def test_corrected_source_does_not_subtract_twice(setup):
    settings, layer = setup
    layer = {**layer, "background_image": layer["image"]}
    corrected = s.settings_for(
        layer,
        settings["table"],
        {
            "source": "corrected",
            "column": "phenotype_ref1",
            "destination": "phenotype_ref1",
            "backgrounds": {"M0": 50000, "M1": 50000},
        },
        [],
    )
    assert corrected["backgrounds"] == {} and corrected["corrected"]
    assert s.run(corrected, label=5)["choices"][0]["phenotype"] == "T"


def test_scan_polling_skips_cells_already_visited(setup):
    settings, _ = setup
    deadline = time.monotonic() + 20
    result = s.request(settings, scan=True)
    while result["status"] == "preparing" and time.monotonic() < deadline:
        time.sleep(0.02)
        result = s.request(settings, scan=True)
    assert result["label"] == 6
    assert s.request(settings, scan=True, exclude=[6])["label"] == 5
    assert s.request(settings, scan=True, exclude=[6, 5])["label"] is None


def test_accepted_labels_keep_training_details_and_retry_identity(setup):
    settings, _ = setup
    suggestion = s.run(settings, label=5)
    table = settings["table"]
    result = reference.save(
        table,
        "roi",
        "phenotype_ref1",
        {5: "T"},
        "accepted-cell-five",
        source="accepted",
        provenance=suggestion["provenance"],
    )
    assert result["suggestion"]["column"] == "phenotype_ref1"
    assert reference.review(table, "roi", "phenotype_ref1")["items"][0]["source"] == "accepted"
    assert (
        reference.save(
            table,
            "roi",
            "phenotype_ref1",
            {5: "T"},
            "accepted-cell-five",
            source="accepted",
            provenance=suggestion["provenance"],
        )
        == result
    )


def test_edits_during_prediction_do_not_return_old_suggestions(setup, monkeypatch):
    settings, _ = setup
    original = s.Model.predict

    def edited(self, values, ids):
        reference.save(settings["table"], "roi", "phenotype_ref1", {6: "B"}, "changed-during-query")
        return original(self, values, ids)

    monkeypatch.setattr(s.Model, "predict", edited)
    with pytest.raises(ValueError, match="changed"):
        s.run(settings, label=5)


def test_cancellation_stops_background_work(setup, monkeypatch):
    import threading

    settings, _ = setup
    started, proceed = threading.Event(), threading.Event()

    def slow(settings, *, progress, **kwargs):
        started.set()
        proceed.wait(5)
        progress("Checking cancellation")

    monkeypatch.setattr(s, "run", slow)
    assert s.request(settings, scan=True)["status"] == "preparing"
    assert started.wait(5)
    assert s.request(settings, scan=True, cancel=True)["status"] == "cancelled"
    proceed.set()
    deadline = time.monotonic() + 5
    while s.request(settings, scan=True)["status"] == "preparing" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert s.request(settings, scan=True)["status"] == "cancelled"
