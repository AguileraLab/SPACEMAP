"""SPACEMAP's cyCombine gives R's result: the same random numbers, SOM groups and corrected values."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import cycombine
from spacemap.batch_correction_utils import BatchCorrector

REFERENCE = json.loads((Path(__file__).parent / "cycombine_r_reference.json").read_text())


def test_r_random_numbers():
    # R: set.seed(473); runif(3); set.seed(473); sample(1:100000, 5); set.seed(9); sample(1:20000000, 3)
    random = cycombine.RRandom(473)
    assert [random.unif() for _ in range(3)] == [0.76718152617104352, 0.37514612032100558, 0.50240290630608797]
    assert cycombine.RRandom(473).sample(100000, 5).tolist() == [24586, 86753, 86613, 36499, 93877]
    assert cycombine.RRandom(9).sample(20000000, 3).tolist() == [12191285, 17119035, 155276]


def test_the_same_groups_and_values_as_r_cycombine():
    values, batch = np.array(REFERENCE["values"]), np.array(REFERENCE["batch"])
    for mode, covar in (("plain", None), ("covar", np.array(REFERENCE["condition"]))):
        expected = REFERENCE[mode]
        grid = expected["grid"]
        corrected, labels, summary = cycombine.batch_correct(values, batch, covar=covar, xdim=grid, ydim=grid)
        assert labels.tolist() == expected["labels"]
        r = np.array([[np.nan if v is None else v for v in row] for row in expected["corrected"]])
        finite = np.isfinite(r).all(axis=1)
        assert np.abs(corrected[finite] - r[finite]).max() < 1e-12
        # Where R gives missing values (a group ComBat cannot estimate), the measured values are kept.
        assert (
            np.array_equal(corrected[~finite], values[~finite]) and summary["cells_kept_as_measured"] == (~finite).sum()
        )
        assert (summary["covar_used"] > 0) == (covar is not None)


def _two_batches(seed=0, n=4000):
    """Two cell types; batch 2 reads every marker 0.5 brighter."""
    rng = np.random.default_rng(seed)
    kind, batch = rng.integers(0, 2, n), rng.integers(0, 2, n)
    means = np.array([[1.0, 3.0, 0.5], [3.0, 1.0, 2.5]])
    values = np.sinh(means[kind] + rng.normal(0, 0.3, (n, 3)) + 0.5 * batch[:, None]) * 5
    frame = pd.DataFrame(values, columns=["A", "B", "C"])
    frame["batch"] = np.array(["b1", "b2"])[batch]
    return frame, kind


def test_both_methods_remove_a_shift_without_widening_cell_types():
    frame, kind = _two_batches()
    before = np.arcsinh(frame[["A", "B", "C"]] / 5).to_numpy()
    for mode in ("cycombine", "monotonic"):
        corrected = BatchCorrector(["A", "B", "C"], correction_mode=mode, verbose=False).correct_batch_effects(
            frame.copy()
        )
        after = corrected[["A_corr", "B_corr", "C_corr"]].to_numpy()
        second = frame["batch"].to_numpy() == "b2"
        for k in (0, 1):
            gap_before = before[second & (kind == k)].mean(axis=0) - before[~second & (kind == k)].mean(axis=0)
            gap_after = after[second & (kind == k)].mean(axis=0) - after[~second & (kind == k)].mean(axis=0)
            assert np.all(np.abs(gap_after) < 0.25 * np.abs(gap_before)), (mode, gap_after)
            # K9: a cell type's spread within a batch stays about the same (the old monotonic scaling widened it).
            spread = after[~second & (kind == k)].std(axis=0) / before[~second & (kind == k)].std(axis=0)
            assert np.all((spread > 0.8) & (spread < 1.2)), (mode, spread)
        assert "som_cluster" in corrected


def test_rank_uses_average_ties_and_each_batch_size():
    # R: rank(c(1, 1, 4, 8)) / 4; rank(c(7, 2)) / 2.
    values = np.array([[1, 0], [1, 0], [4, 0], [8, 0], [7, 0], [2, 0]], dtype=float)
    actual = cycombine.normalize(values, ["a"] * 4 + ["b"] * 2, norm_method="rank")
    np.testing.assert_array_equal(actual[:, 0], [0.375, 0.375, 0.75, 1.0, 1.0, 0.5])
    np.testing.assert_array_equal(actual[:, 1], 0)
    np.testing.assert_array_equal(cycombine.normalize(values, ["a"] * 6, "none"), values)
    with pytest.raises(ValueError, match="normalization"):
        cycombine.normalize(values, ["a"] * 6, "unknown")


def test_wrapper_honours_rank_instead_of_silently_scaling():
    frame, _ = _two_batches(n=300)
    raw = frame[["A", "B", "C"]].to_numpy()
    expected, labels, _ = cycombine.batch_correct(np.arcsinh(raw / 5), frame.batch.to_numpy(),
                                                 norm_method="rank", xdim=3, ydim=3)
    corrector = BatchCorrector(["A", "B", "C"], normalization_method="rank", som_grid_size=(3, 3), verbose=False)
    result = corrector.correct_batch_effects(frame.copy())
    np.testing.assert_array_equal(result.som_cluster, labels)
    np.testing.assert_allclose(result[["A_corr", "B_corr", "C_corr"]], expected)
    assert corrector.run_summary["som"]["normalization"] == "rank"


def test_small_som_input_has_an_actionable_error():
    with pytest.raises(ValueError, match="smaller SOM grid"):
        cycombine.create_som(np.zeros((3, 2)), xdim=2, ydim=2)


def test_confounded_covariates_and_single_batch_groups_are_reported():
    rng = np.random.default_rng(29)
    values = rng.normal(size=(90, 4))
    batches = np.array(["a"] * 40 + ["b"] * 50)
    labels = np.array([1] * 80 + [2] * 10)
    corrected, summary = cycombine.correct_data(values, batches, labels, covar=batches)
    assert summary["covar_omitted"] == 1 and summary["covar_used"] == 0
    assert summary["one_batch"] == 1 and summary["one_batch_cells"] == 10
    np.testing.assert_array_equal(corrected[-10:], values[-10:])
    assert np.isfinite(corrected).all()


def test_post_qc_compares_on_the_same_scale_with_custom_cofactor():
    from spacemap.batch_correction_utils import detect_overcorrected_cells
    frame, _ = _two_batches(n=100)
    for marker in ('A', 'B', 'C'):
        frame[marker + '_corr'] = np.arcsinh(frame[marker] / 150)
    found = detect_overcorrected_cells(frame, ['A', 'B', 'C'], batch_col='batch', cofactor=150)
    assert (found.correction_distance == 0).all()
    assert not found.overcorrected.any()
