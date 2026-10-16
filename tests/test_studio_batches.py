"""Studio's batch correction checks give Tab 6's own measures."""

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.studio import batches


@pytest.fixture
def table(tmp_path):
    """Four batches whose markers are shifted, and a correction that takes the shift out."""
    rng = np.random.default_rng(1)
    parts = []
    for index, (name, cells) in enumerate([("S1", 900), ("S2", 400), ("S3", 700), ("S4", 120)]):
        frame = pd.DataFrame({"batch_id": name, "label": np.arange(cells)})
        for marker, scale in (("CD8", 40.0), ("PanCK", 300.0)):
            raw = rng.gamma(2.0, scale, cells) * (1 + 0.6 * index)
            frame[marker] = raw
            corrected = np.arcsinh(raw / 5.0)
            frame[marker + "_corr"] = corrected - np.median(corrected) + 3.0
        parts.append(frame)
    path = tmp_path / "combined.csv"
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return path


def test_the_sample_keeps_every_batch(table):
    found = batches.sample(table, "batch_id")
    counts = found["frame"]["batch_id"].value_counts()
    assert set(counts.index) == {"S1", "S2", "S3", "S4"} and found["markers"] == ["CD8", "PanCK"]
    assert counts["S4"] == 120 and counts.max() <= 1.2 * batches.PER_BATCH
    assert found["cells"] == 2120


def test_the_measures_are_tab6s(table):
    from spacemap.batch_correction_utils import calculate_emd_reduction, calculate_mad_score

    frame = batches.sample(table, "batch_id")["frame"]
    overall, per_marker = batches.measures(frame, ["CD8", "PanCK"], "batch_id", 5.0)
    assert overall["emd_reduction"] == pytest.approx(
        calculate_emd_reduction(frame, frame, ["CD8", "PanCK"], batch_col="batch_id", cofactor=5.0), abs=1e-4
    )
    assert overall["mad"] == pytest.approx(
        calculate_mad_score(frame, frame, ["CD8", "PanCK"], batch_col="batch_id", cofactor=5.0), abs=1e-4
    )
    for item in per_marker:
        assert item["emd_reduction"] == pytest.approx(
            calculate_emd_reduction(frame, frame, [item["marker"]], batch_col="batch_id", cofactor=5.0), abs=1e-4
        )
    assert overall["emd_reduction"] > 0.5


def test_the_check_reports_batches_measures_and_a_heatmap(table):
    config = {
        "tab_configurations": {"tab6_batch_correction": {"bc_config": {"method": "Combat", "arcsinh_cofactor": 5}}}
    }
    found = batches.check(table, config)
    assert found["batch_column"] == "batch_id" and found["method"] == "Combat" and found["cells"] == 2120
    assert [item["name"] for item in found["batches"]] == ["S1", "S2", "S3", "S4"]
    heatmap = found["heatmap"]
    assert len(heatmap["before"]) == 4 and len(heatmap["before"][0]) == 2 and heatmap["markers"] == ["CD8", "PanCK"]
    # Before, the batch medians climb with the shift; after, they are level.
    before = np.array(heatmap["before"])[:, 0]
    after = np.array(heatmap["after"])[:, 0]
    assert np.all(np.diff(before) > 0) and np.ptp(after) < 0.2 * np.ptp(before)
    with pytest.raises(ValueError, match="no 'plate' column"):
        batches.check(
            table, {"tab_configurations": {"tab6_batch_correction": {"bc_config": {"batch_id_col": "plate"}}}}
        )


def test_histograms_show_each_batch_before_and_after_on_one_scale(table):
    config = {"tab_configurations": {"tab6_batch_correction": {"bc_config": {"arcsinh_cofactor": 5}}}}
    found = batches.check(table, config)["histograms"]
    assert found["units"] == ["S1", "S2", "S3", "S4"] and found["bins"] == batches.HISTOGRAM_BINS
    cd8 = found["markers"]["CD8"]
    before, after = np.array(cd8["before"]), np.array(cd8["after"])
    assert before.shape == (4, batches.HISTOGRAM_BINS) and np.allclose(before.sum(axis=1), 1, atol=1e-3)
    # Before, each batch's middle sits further right; after the correction the middles line up.
    middle = lambda shares: (np.cumsum(shares, axis=1) < 0.5).sum(axis=1)  # noqa: E731
    assert np.all(np.diff(middle(before)) > 0) and np.ptp(middle(after)) <= 1


def test_checks_use_saved_run_settings_and_direct_parquet(table):
    import json
    frame = pd.read_csv(table)
    target = table.with_name('corrected.parquet')
    frame.rename(columns={'CD8_corr': 'CD8_fixed', 'PanCK_corr': 'PanCK_fixed'}).to_parquet(target, index=False)
    target.with_suffix('.json').write_text(json.dumps({'rois': ['S1'], 'mode': 'cycombine',
        'config': {'arcsinh_cofactor': 5, 'corrected_suffix': '_fixed', 'batch_id_col': 'batch_id'}}))
    # The next run's settings must not change the interpretation of this result.
    config = {'tab_configurations': {'tab6_batch_correction': {'bc_config': {'arcsinh_cofactor': 150, 'corrected_suffix': '_wrong'}}}}
    found = batches.check(target, config)
    expected = batches.check(table, {})
    assert found['cofactor'] == 5 and found['suffix'] == '_fixed'
    assert found['overall'] == expected['overall']
    assert found['method'] == 'cyCombine (Python port)'


def test_diagnostics_compare_matched_finite_cells_only(table):
    frame = pd.read_csv(table)
    excluded = frame.iloc[[0]].copy()
    excluded['CD8'] = 1e20
    excluded['CD8_corr'] = np.nan
    invalid = excluded.copy()
    invalid['CD8_corr'] = np.inf
    augmented = pd.concat([frame, excluded, invalid], ignore_index=True)
    assert batches.measures(augmented, ['CD8'], 'batch_id', 5) == batches.measures(frame, ['CD8'], 'batch_id', 5)
    assert batches.histograms(augmented, ['CD8'], 'batch_id', 5) == batches.histograms(frame, ['CD8'], 'batch_id', 5)
