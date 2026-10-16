"""Live RESOLVE must give exactly Tab 7's phenotypes for the same thresholds."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from spacemap.resolve_live import LiveResolve, Panel, Settings, hierarchy

MARKERS = ["CD45", "CD3", "CD4", "CD8", "FOXP3", "CD20", "CD68", "CD163", "PanCK"]
PANEL = Panel(
    definitions={
        "CD4 T": [["CD45", "CD3", "CD4"]],
        "Treg": [["CD45", "CD3", "CD4", "FOXP3"]],
        "CD8 T": [["CD45", "CD3", "CD8"], ["CD3", "CD8"]],
        "B": [["CD45", "CD20"]],
        "Macrophage": [["CD45", "CD68"], ["CD68", "CD163"]],
        "Tumour": [["PanCK"]],
        "Needs a missing marker": [["CD45", "NotMeasured"]],
    },
    supportive={"Macrophage": ["CD163"], "B": ["CD45"]},
    exclusions={"CD8 T": ["CD4"], "Tumour": ["CD45"]},
    unclassified_label="Others",
)


def tab7(frame: pd.DataFrame, thresholds: dict, settings: Settings) -> pd.Series:
    """Tab 7's own RESOLVE path with a predefined threshold table (resolve_clustering, run section)."""
    from spacemap.ph_thres_utils import (
        apply_exclusion_markers,
        apply_supportive_markers,
        assign_phenotypes_binary_tiebreak,
    )
    from spacemap_gui.tabs.tab7_phenotyping.resolve_clustering import _apply_thresholds_from_table

    frame = frame[~frame[MARKERS].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)].copy()
    table = pd.DataFrame([{"batch_id": frame["batch_id"].iloc[0], **thresholds}])
    binary, norm = _apply_thresholds_from_table(
        df=frame, thresholds_table=table, selected_markers=MARKERS, batch_col="batch_id",
        norm_flags=settings.norm_flags, norm_order=list(settings.norm_order), method_tag="resolve",
        tie_score_transform=settings.tie_score_transform, tie_score_cofactor=settings.tie_score_cofactor)
    definitions = PANEL.definitions
    assigned = assign_phenotypes_binary_tiebreak(
        df_processed=pd.concat([binary, norm], axis=1), bin_marker_cols=list(binary.columns),
        norm_marker_cols=list(norm.columns), phenotype_col_name="p", phenotype_definitions=definitions,
        phenotype_hierarchy=hierarchy(definitions))["p"]
    both = pd.concat([binary, norm], axis=1)
    assigned = apply_supportive_markers(assigned, both, list(binary.columns), list(norm.columns), definitions,
                                        PANEL.supportive)
    assigned = apply_exclusion_markers(assigned, both, list(norm.columns), definitions, PANEL.exclusions, reassign=True)
    return assigned.replace({"unclassified": PANEL.unclassified_label})


def cells(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({m: rng.gamma(1.5, 60, n) for m in MARKERS})
    frame.loc[rng.choice(n, 5, replace=False), "CD8"] = np.nan  # left out, as Tab 7 does
    frame["batch_id"] = "s1r1"
    return frame


SETTINGS = [
    Settings(),
    Settings(tie_score_transform="asinh_difference", tie_score_cofactor=5.0),  # PDAC's settings
    Settings(norm_flags=(True, False, True, False), tie_score_transform="log1p_difference"),
    Settings(norm_order=("Z2", "Z1", "CDF", "NLog")),
]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("settings", SETTINGS, ids=["default", "asinh", "z1-cdf-log1p", "z2-first"])
def test_same_phenotype_as_tab7_for_every_cell(seed, settings):
    frame = cells(3000, seed)
    rng = np.random.default_rng(100 + seed)
    thresholds = {m: float(rng.uniform(40, 140)) for m in MARKERS}
    expected = tab7(frame, thresholds, settings)
    live = LiveResolve(frame[MARKERS].to_numpy(), MARKERS, PANEL, settings).assign(thresholds)
    got = pd.Series(live.phenotypes, index=frame.index)
    assert got.isna().sum() == 5  # the cells with a missing marker
    kept = got.dropna()
    assert (kept.reindex(expected.index) == expected).all(), (kept.reindex(expected.index) != expected).sum()
    assert sum(live.counts.values()) == len(expected)


def test_panel_file_and_hierarchy():
    payload = {"criteria_cols": ["Criteria 1", "Criteria 2"], "table": [
        {"Phenotype": "CD4 T", "Criteria 1": ["CD3", "CD4"], "Criteria 2": [], "Supportive": [], "Must not have": ["CD8"]},
        {"Phenotype": "Treg", "Criteria 1": ["CD3", "CD4", "FOXP3"], "Criteria 2": [], "Supportive": ["CD25"]},
        {"Phenotype": "", "Criteria 1": ["X"]}]}
    panel = Panel.from_payload(payload)
    assert panel.definitions == {"CD4 T": [["CD3", "CD4"]], "Treg": [["CD3", "CD4", "FOXP3"]]}
    assert panel.exclusions == {"CD4 T": ["CD8"]} and panel.supportive == {"Treg": ["CD25"]}
    assert hierarchy(panel.definitions) == {"CD4 T": None, "Treg": "CD4 T"}


def test_why_a_cell_got_its_phenotype():
    frame = cells(500, 3)
    thresholds = {m: 80.0 for m in MARKERS}
    engine = LiveResolve(frame[MARKERS].to_numpy(), MARKERS, PANEL, Settings())
    result = engine.assign(thresholds)
    cell = int(np.flatnonzero(pd.notna(result.phenotypes))[0])
    why = engine.explain(cell, thresholds, result)
    assert why["phenotype"] == result.phenotypes[cell]
    cd8 = next(p for p in why["phenotypes"] if p["phenotype"] == "CD8 T")
    value = frame["CD8"].iloc[cell]
    assert cd8["criteria"][1]["met"] == (frame["CD3"].iloc[cell] > 80 and value > 80)
    missing = next(p for p in why["phenotypes"] if p["phenotype"] == "Needs a missing marker")
    assert "NotMeasured" in missing["criteria"][0]["failing"]
    left_out = int(np.flatnonzero(pd.isna(result.phenotypes))[0])
    assert engine.explain(left_out, thresholds, result)["left_out"]


def test_fast_enough_to_follow_a_dragged_threshold():
    frame = cells(100_000, 4)
    engine = LiveResolve(frame[MARKERS].to_numpy(), MARKERS, PANEL, Settings(tie_score_transform="asinh_difference"))
    thresholds = {m: 80.0 for m in MARKERS}
    engine.assign(thresholds)
    started = time.perf_counter()
    for value in (60.0, 70.0, 90.0, 100.0, 110.0):
        thresholds["CD8"] = value
        engine.assign(thresholds)
    per_update = (time.perf_counter() - started) / 5
    # Loose on purpose (shared nodes). Measured on PDAC: s1r2 (158,463 cells) 17 ms and s16r1
    # (1,382,067 cells) 141 ms per change, the same phenotype as Tab 7 for every cell.
    assert per_update < 0.15, f"{per_update * 1000:.0f} ms per update"


@pytest.mark.parametrize("settings", SETTINGS[:2], ids=["default", "asinh"])
def test_changing_thresholds_one_by_one_gives_the_same_as_starting_fresh(settings):
    # Each change redoes only the changed marker's columns; any stale part would show here.
    frame = cells(4000, 7)
    rng = np.random.default_rng(7)
    thresholds = {m: float(rng.uniform(40, 140)) for m in MARKERS}
    engine = LiveResolve(frame[MARKERS].to_numpy(), MARKERS, PANEL, settings)
    engine.assign(thresholds)
    for step in range(25):
        marker = MARKERS[int(rng.integers(len(MARKERS)))]
        thresholds[marker] = float(rng.uniform(30, 160)) if step % 5 else thresholds[marker]  # sometimes no change
        live = engine.assign(thresholds)
        fresh = LiveResolve(frame[MARKERS].to_numpy(), MARKERS, PANEL, settings).assign(thresholds)
        assert list(live.phenotypes) == list(fresh.phenotypes), (step, marker)
        assert live.counts == fresh.counts
        assert (live.fired == fresh.fired).all() and (live.positive == fresh.positive).all()
    expected = tab7(frame, thresholds, settings)
    kept = pd.Series(engine.assign(thresholds).phenotypes, index=frame.index).dropna()
    assert (kept.reindex(expected.index) == expected).all()
