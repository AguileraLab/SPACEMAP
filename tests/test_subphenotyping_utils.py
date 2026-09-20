"""Tests for the objective sub-phenotyping pipeline."""

import numpy as np
import pandas as pd
import pytest

from spacemap.ph_subpheno_utils import (
    BIMODAL_BC,
    COLLAPSED_SD_RATIO,
    build_feature_matrix,
    cluster_stability,
    marker_reference_stats,
    recommend_markers,
    score_markers,
    subcluster,
    validate_subclusters,
)


@pytest.fixture
def cohort():
    """Synthetic cohort with a known answer.

    Inside the "Target" phenotype there are two real states separated by
    ``FunctionalA``. The other markers are built to reproduce each failure mode
    the scoring is supposed to catch.
    """
    rng = np.random.default_rng(0)
    n, n_other = 4000, 4000
    batches = np.array([f"b{i%8}" for i in range(n)])

    target = pd.DataFrame({
        # real within-phenotype split: two well-separated states
        "FunctionalA": np.where(rng.random(n) < 0.4, rng.normal(3, 0.4, n), rng.normal(0, 0.4, n)),
        # graded biology
        "FunctionalB": rng.normal(2, 1.0, n),
        # the gating marker: high, but genuinely graded -> must be kept
        "GateMarker": rng.normal(4, 0.9, n),
        # another lineage's marker: flat at zero -> noise if standardised
        "ForeignFlat": rng.normal(0, 0.02, n),
        # spread driven entirely by which batch the cell came from
        "BatchDriven": np.array([{"b%d" % i: i * 2.0 for i in range(8)}[b] for b in batches])
        + rng.normal(0, 0.05, n),
        # foreign marker that switches on/off -> looks like spillover
        "ForeignBimodal": np.where(rng.random(n) < 0.2, rng.normal(4, 0.3, n), rng.normal(0, 0.3, n)),
        "DAPI": rng.normal(3, 1.0, n),
    })
    target["phenotype"] = "Target"
    target["batch_id"] = batches

    other = pd.DataFrame({
        "FunctionalA": rng.normal(1, 1.0, n_other),
        "FunctionalB": rng.normal(1, 1.0, n_other),
        "GateMarker": rng.normal(0, 0.5, n_other),      # low elsewhere -> Target owns it
        "ForeignFlat": rng.normal(4, 1.5, n_other),     # high elsewhere -> foreign to Target
        "BatchDriven": rng.normal(7, 4.0, n_other),
        "ForeignBimodal": rng.normal(5, 1.0, n_other),  # high elsewhere -> foreign to Target
        "DAPI": rng.normal(3, 1.0, n_other),
    })
    other["phenotype"] = "Other"
    other["batch_id"] = np.array([f"b{i%8}" for i in range(n_other)])

    # A third population, so "how concentrated is this marker across phenotypes"
    # is measurable: with only two groups the median sits halfway and no marker
    # can look lineage-restricted.
    bystander = pd.DataFrame({
        "FunctionalA": rng.normal(1, 1.0, n_other),
        "FunctionalB": rng.normal(1, 1.0, n_other),
        "GateMarker": rng.normal(0, 0.5, n_other),
        "ForeignFlat": rng.normal(0, 0.5, n_other),
        "BatchDriven": rng.normal(7, 4.0, n_other),
        "ForeignBimodal": rng.normal(0, 0.5, n_other),   # restricted to "Other"
        "DAPI": rng.normal(3, 1.0, n_other),
    })
    bystander["phenotype"] = "Bystander"
    bystander["batch_id"] = np.array([f"b{i%8}" for i in range(n_other)])

    df = pd.concat([target, other, bystander], ignore_index=True)
    markers = ["FunctionalA", "FunctionalB", "GateMarker", "ForeignFlat",
               "BatchDriven", "ForeignBimodal", "DAPI"]
    return df, markers


@pytest.fixture
def scored(cohort):
    df, markers = cohort
    reference = marker_reference_stats(df, markers, "phenotype")
    subset = df[df.phenotype == "Target"]
    return score_markers(subset, markers, reference=reference,
                         batch_col="batch_id", phenotype="Target"), subset, markers


# ---------------------------------------------------------------------------
# Marker discovery
# ---------------------------------------------------------------------------


def test_flat_foreign_marker_is_detected_as_collapsed(scored):
    scores, _, _ = scored
    row = scores.set_index("marker").loc["ForeignFlat"]

    assert row.sd_ratio < COLLAPSED_SD_RATIO, "a marker flat inside the phenotype must collapse"
    assert row.foreign, "it belongs to another lineage"


def test_batch_driven_marker_is_attributed_to_batch(scored):
    scores, _, _ = scored
    row = scores.set_index("marker").loc["BatchDriven"]

    assert row.batch_eta2 > 0.9, "spread is entirely between batches"
    assert row["score"] < scores.set_index("marker").loc["FunctionalA"]["score"]


def test_real_split_marker_ranks_above_noise(scored):
    scores, _, _ = scored
    ranked = scores.marker.tolist()

    assert ranked.index("FunctionalA") < ranked.index("ForeignFlat")
    assert scores.set_index("marker").loc["FunctionalA"].bimodality > BIMODAL_BC


def test_gating_marker_is_kept_when_it_still_varies(scored):
    """The whole point: a marker used to define the phenotype is not auto-dropped."""
    scores, _, _ = scored
    recommended = recommend_markers(scores)
    row = recommended.set_index("marker").loc["GateMarker"]

    assert row.recommended, "a graded gating marker carries real within-phenotype biology"
    assert row.flag == ""


def test_recommendation_rejects_each_failure_mode_with_a_reason(scored):
    scores, _, _ = scored
    rec = recommend_markers(scores).set_index("marker")

    assert rec.loc["DAPI"].flag == "excluded marker"
    assert rec.loc["ForeignFlat"].flag == "flat in this phenotype"
    assert rec.loc["BatchDriven"].flag == "batch-driven"
    assert rec.loc["ForeignBimodal"].flag == "possible spillover"
    assert rec.loc["FunctionalA"].recommended
    assert rec.loc["FunctionalB"].recommended


def test_universal_functional_marker_is_not_called_spillover(scored):
    """Ki67-like case: bimodal and lower here than elsewhere, but carried by every
    cell type, so it is real biology rather than contamination from a neighbour."""
    scores, _, _ = scored
    rec = recommend_markers(scores).set_index("marker")

    # FunctionalA is bimodal inside Target and present across all phenotypes.
    assert rec.loc["FunctionalA"].lineage_spec < 1.5
    assert rec.loc["FunctionalA"].flag != "possible spillover"


def test_marker_cap_keeps_the_most_informative(scored):
    scores, _, _ = scored
    rec = recommend_markers(scores, max_markers=2)

    assert rec.recommended.sum() == 2
    assert "below marker cap" in set(rec.flag)


def test_scoring_without_a_batch_column_still_works(scored):
    _, subset, markers = scored
    scores = score_markers(subset, markers)

    assert (scores.batch_eta2 == 0).all()
    assert scores["score"].notna().all()


def test_scoring_needs_enough_cells(scored):
    _, subset, markers = scored
    with pytest.raises(ValueError):
        score_markers(subset.head(3), markers)


# ---------------------------------------------------------------------------
# Feature space
# ---------------------------------------------------------------------------


def test_feature_matrix_removes_batch_offsets(cohort):
    df, _ = cohort
    subset = df[df.phenotype == "Target"]

    matrix = build_feature_matrix(subset, ["BatchDriven"], batch_col="batch_id")

    # BatchDriven is a pure per-batch constant; within-batch standardisation must
    # flatten the between-batch structure that would otherwise form clusters.
    per_batch_means = pd.Series(matrix[:, 0]).groupby(subset.batch_id.values).mean()
    assert per_batch_means.abs().max() < 0.1


def test_feature_matrix_is_finite_and_clipped(cohort):
    df, markers = cohort
    subset = df[df.phenotype == "Target"].copy()
    subset.loc[subset.index[:5], "FunctionalB"] = np.nan

    matrix = build_feature_matrix(subset, markers, batch_col="batch_id", clip=3.0)

    assert np.isfinite(matrix).all()
    assert np.abs(matrix).max() <= 3.0


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,params", [
    ("Leiden", {"n_neighbors": 30, "resolution": 0.5, "n_pcs": 4, "n_meta_clusters": 2}),
    ("SOM", {"grid": 6, "n_meta_clusters": 2, "iterations": 2000}),
    ("Vortex", {"k_neighbors": 30, "min_cluster_size": 50, "n_meta_clusters": 2}),
])
def test_every_method_recovers_the_planted_split(scored, method, params):
    _, subset, _ = scored
    matrix = build_feature_matrix(subset, ["FunctionalA", "FunctionalB"], batch_col="batch_id")

    labels = subcluster(matrix, method, params, seed=0, subsample_size=None)

    assert len(labels) == len(matrix)
    assert len(np.unique(labels)) >= 2
    # The planted state must map onto the partition rather than being split at random.
    truth = (subset.FunctionalA.values > 1.5).astype(int)
    from sklearn.metrics import adjusted_mutual_info_score
    assert adjusted_mutual_info_score(truth, labels) > 0.5, f"{method} lost the real structure"


def test_unknown_method_is_rejected(scored):
    _, subset, _ = scored
    matrix = build_feature_matrix(subset, ["FunctionalA"], batch_col="batch_id")
    with pytest.raises(ValueError, match="Unknown clustering method"):
        subcluster(matrix, "kmeans", {})


def test_sketching_labels_every_cell(scored):
    _, subset, _ = scored
    matrix = build_feature_matrix(subset, ["FunctionalA", "FunctionalB"], batch_col="batch_id")

    labels = subcluster(matrix, "Leiden", {"n_neighbors": 15, "resolution": 0.5, "n_pcs": 2},
                        seed=0, subsample_size=1000)

    assert len(labels) == len(matrix)
    assert not pd.isna(labels).any()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_flags_a_single_batch_cluster():
    rng = np.random.default_rng(1)
    matrix = rng.normal(0, 1, (1000, 3)).astype(np.float32)
    labels = np.zeros(1000, dtype=int)
    batches = np.array([f"b{i%10}" for i in range(1000)])
    # a "cluster" living entirely in one batch, with a real marker shift
    labels[:200] = 1
    batches[:200] = "b0"
    matrix[:200, 0] += 4

    report = validate_subclusters(labels, matrix, ["m0", "m1", "m2"], batch_labels=batches)
    row = report.set_index("cluster").loc[1]

    assert row.verdict == "review"
    assert "not reproducible across batches" in row.reason
    assert row.top_marker == "m0"


def test_validation_flags_a_cluster_with_no_separating_marker():
    rng = np.random.default_rng(2)
    matrix = rng.normal(0, 1, (1000, 3)).astype(np.float32)
    labels = (np.arange(1000) % 2).astype(int)     # arbitrary split, no marker behind it
    batches = np.array([f"b{i%10}" for i in range(1000)])

    report = validate_subclusters(labels, matrix, ["m0", "m1", "m2"], batch_labels=batches)

    assert (report.verdict == "review").all()
    assert all("no separating marker" in r for r in report.reason)


def test_validation_keeps_a_real_reproducible_cluster():
    rng = np.random.default_rng(3)
    matrix = rng.normal(0, 1, (2000, 3)).astype(np.float32)
    labels = np.zeros(2000, dtype=int)
    labels[:800] = 1
    matrix[:800, 1] += 3
    batches = np.array([f"b{i%10}" for i in range(2000)])

    report = validate_subclusters(labels, matrix, ["m0", "m1", "m2"], batch_labels=batches)

    assert (report.verdict == "keep").all()
    assert report.set_index("cluster").loc[1].top_marker == "m1"
    assert report.cells.sum() == 2000


def test_small_clusters_are_flagged():
    matrix = np.random.default_rng(4).normal(0, 1, (500, 2)).astype(np.float32)
    labels = np.zeros(500, dtype=int)
    labels[:10] = 1
    matrix[:10, 0] += 6

    report = validate_subclusters(labels, matrix, ["m0", "m1"], min_cells=100)

    assert "too few cells" in report.set_index("cluster").loc[1].reason


def test_stability_is_high_for_real_structure_and_low_for_noise():
    rng = np.random.default_rng(5)
    params = {"n_neighbors": 30, "resolution": 0.1, "n_pcs": 2}

    real = np.vstack([rng.normal(-3, 0.5, (600, 2)), rng.normal(3, 0.5, (600, 2))]).astype(np.float32)
    noise = rng.normal(0, 1, (1200, 2)).astype(np.float32)

    stable = cluster_stability(real, "Leiden", params, n_repeats=3, seed=0)
    unstructured = cluster_stability(noise, "Leiden", params, n_repeats=3, seed=0)

    assert stable > 0.7, f"a planted split should be reproducible (got {stable:.2f})"
    # Featureless data collapses to one cluster: reported as "no structure",
    # never as a perfect score.
    assert np.isnan(unstructured) or unstructured < stable
