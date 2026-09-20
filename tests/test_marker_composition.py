"""Tests for the high-expression composition panel in Sub-Phenotyping."""

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.tabs.tab7_phenotyping.marker_composition import (
    METRIC_ENRICHMENT,
    METRIC_POSITIVE,
    METRIC_SHARE,
    apply_filters,
    build_summary_table,
    compute_composition,
    default_phenotype_colors,
    marker_reference_stats,
    normalize_phenotype_labels,
    order_markers,
    profile_values,
)


@pytest.fixture
def cohort():
    """Synthetic cohort with a known answer.

    ``CD8Marker`` is bright only in CD8 cells and ``MacMarker`` only in
    macrophages, so both must come out as the enriched marker of their
    phenotype. ``Ubiquitous`` is bright at random, so its high cells must
    follow the phenotype abundances rather than any one lineage.
    """
    rng = np.random.default_rng(0)
    n = 12000
    phenotypes = rng.choice(
        ["Epithelial", "CD8", "Macrophage", None], n, p=[0.5, 0.2, 0.25, 0.05]
    )
    frame = pd.DataFrame({
        "phenotype": phenotypes,
        "batch_id": rng.choice(["P1", "P2"], n),
        "area": rng.normal(50, 10, n),
    })
    is_cd8 = frame.phenotype == "CD8"
    is_mac = frame.phenotype == "Macrophage"
    frame["CD8Marker"] = rng.normal(0, 1, n) + is_cd8 * 5
    frame["MacMarker"] = rng.normal(0, 1, n) + is_mac * 5
    frame["Ubiquitous"] = rng.normal(0, 1, n)
    frame["Constant"] = 3.0
    # Background is exactly zero in most cells, so the MAD is zero here.
    frame["MostlyZero"] = np.where(rng.random(n) < 0.8, 0.0, rng.normal(4, 1, n))
    return frame


MARKERS = ["CD8Marker", "MacMarker", "Ubiquitous", "Constant", "MostlyZero"]


def _categories(frame):
    return sorted(normalize_phenotype_labels(frame["phenotype"]).unique())


def test_missing_labels_are_one_visible_group(cohort):
    labels = normalize_phenotype_labels(cohort["phenotype"])
    assert "Unassigned" in set(labels)
    assert not labels.isna().any()
    assert "nan" not in set(labels)


def test_composition_finds_the_marker_of_each_phenotype(cohort):
    result = compute_composition(cohort, MARKERS, "phenotype", 1.5, _categories(cohort))

    assert result["share"].loc["CD8Marker", "CD8"] > 95
    assert result["share"].loc["MacMarker", "Macrophage"] > 95
    # Every marker's composition is a percentage of its own high cells.
    for marker in ("CD8Marker", "MacMarker", "Ubiquitous"):
        assert result["share"].loc[marker].sum() == pytest.approx(100.0)
    # A marker bright at random tracks the phenotype abundances.
    assert result["share"].loc["Ubiquitous", "Epithelial"] == pytest.approx(
        result["baseline"]["Epithelial"], abs=6.0
    )
    # A marker with no spread has no high cells rather than a divide-by-zero.
    assert result["n_high"]["Constant"] == 0
    assert result["share"].loc["Constant"].sum() == 0


def test_robust_scaling_falls_back_when_the_mad_is_zero(cohort):
    stats = marker_reference_stats(cohort, MARKERS, robust=True)
    centre, scale = stats["MostlyZero"]
    assert np.isfinite(centre) and np.isfinite(scale) and scale > 0

    result = compute_composition(
        cohort, MARKERS, "phenotype", 1.5, _categories(cohort), stats=stats, robust=True
    )
    assert result["n_high"]["MostlyZero"] > 0


def test_shared_stats_keep_one_scale_across_strata(cohort):
    stats = marker_reference_stats(cohort, MARKERS, robust=False)
    subset = cohort[cohort.batch_id == "P1"]
    shared = compute_composition(
        subset, MARKERS, "phenotype", 1.5, _categories(cohort), stats=stats
    )
    own = compute_composition(subset, MARKERS, "phenotype", 1.5, _categories(cohort))

    assert shared["n_cells"] == len(subset)
    # Both keep every phenotype of the parent cohort so figures stay comparable.
    assert list(shared["counts"].columns) == _categories(cohort)
    assert shared["stats"]["CD8Marker"] == stats["CD8Marker"]
    assert own["stats"]["CD8Marker"] != stats["CD8Marker"]


def test_summary_table_ranks_by_enrichment_and_respects_the_floor(cohort):
    result = compute_composition(cohort, MARKERS, "phenotype", 1.5, _categories(cohort))
    table = build_summary_table(result, top_n=3, metric=METRIC_ENRICHMENT, min_cells=10)

    top = table[table.Phenotype == "CD8"].iloc[0]
    assert top["_marker"] == "CD8Marker"
    assert top[METRIC_ENRICHMENT] > 1.5
    assert top[METRIC_POSITIVE] > 50
    assert (table["High cells (n)"] >= 10).all()
    assert table.groupby("Phenotype").size().max() <= 3

    # A floor no pair can clear yields an empty table, not an exception.
    assert build_summary_table(result, 3, METRIC_SHARE, min_cells=10**9).empty


def test_marker_order_groups_by_dominant_phenotype(cohort):
    result = compute_composition(cohort, MARKERS, "phenotype", 1.5, _categories(cohort))

    assert order_markers(result, MARKERS, "Panel order") == MARKERS
    assert order_markers(result, MARKERS, "Alphabetical") == sorted(MARKERS, key=str.lower)
    grouped = order_markers(result, MARKERS, "Grouped by dominant phenotype")
    assert grouped.index("CD8Marker") < grouped.index("MacMarker")
    assert grouped[-1] == "Constant"


def test_filters_are_applied_in_order(cohort):
    filtered, notes = apply_filters(cohort, [
        {"column": "batch_id", "kind": "values", "values": ["P1"]},
        {"column": "area", "kind": "range", "low": 40.0, "high": 60.0},
        {"column": "not_a_column", "kind": "values", "values": ["x"]},
    ])
    assert set(filtered.batch_id.unique()) == {"P1"}
    assert filtered.area.between(40.0, 60.0).all()
    assert len(notes) == 3
    assert "not_a_column" in notes[-1]


def test_phenotype_colors_follow_the_house_palette(cohort):
    colors = default_phenotype_colors(["Epithelial", "CD8", "Macrophage", "Novel State"])
    assert colors["Epithelial"] == "#800080"
    assert colors["CD8"] == "#DC143C"
    # An unlisted phenotype still gets a colour, and no two labels share one.
    assert colors["Novel State"].startswith("#")
    assert len(set(colors.values())) == 4


def test_continuous_columns_are_flagged_for_stratification():
    assert profile_values([str(v) for v in range(200)])["looks_continuous"]
    assert not profile_values(["P1", "P2", "P3"])["looks_continuous"]
    assert not profile_values([str(v) for v in range(5)])["looks_continuous"]
