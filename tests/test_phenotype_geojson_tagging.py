"""Tests for the Phenotype GeoJSON tagging pipeline (acc_utils + TOOLS helpers)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from spacemap.acc_utils import (
    DEFAULT_UNASSIGNED_LABEL,
    annotate_geojson_with_phenotypes,
    build_agreement_labels,
    build_color_mapping,
    default_palette,
    feature_label,
    generate_png_from_geojson,
    missing_label_mask,
)


GRID = 6
CELL_SIZE = 10.0
PHENOTYPES = ["T cell", "CAF", "Tumor"]


def _square(x0: float, y0: float, size: float = 8.0) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]],
    }


@pytest.fixture
def segmentation(tmp_path):
    """A 6x6 grid of segmented cells; only the first 4 rows carry a CSV centroid."""
    features = [
        {"type": "Feature", "properties": {"id": f"{i}_{j}"}, "geometry": _square(i * CELL_SIZE, j * CELL_SIZE)}
        for i in range(GRID)
        for j in range(GRID)
    ]
    geojson_path = tmp_path / "S1_cell_boundaries.geojson"
    geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    rows = [
        {
            "batch_id": "S1",
            "x_centroid": i * CELL_SIZE + 4,
            "y_centroid": j * CELL_SIZE + 4,
            "phenotype_leiden": PHENOTYPES[(i + j) % len(PHENOTYPES)],
            "region": "core" if i < 3 else "edge",
            "area": float(10 * i + j),
        }
        for i in range(GRID)
        for j in range(4)
    ]
    csv_path = tmp_path / "unified.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"geojson": str(geojson_path), "csv": str(csv_path), "dir": tmp_path}


def _annotate(segmentation, name="out", **kwargs):
    stats: dict = {}
    output_path = str(segmentation["dir"] / f"{name}.geojson")
    kwargs.setdefault("generate_png", False)
    ok = annotate_geojson_with_phenotypes(
        csv_path=segmentation["csv"],
        geojson_path=segmentation["geojson"],
        output_path=output_path,
        phenotype_col="phenotype_leiden",
        stats=stats,
        **kwargs,
    )
    return ok, stats, output_path


def test_unmatched_polygons_are_dropped_by_default(segmentation):
    ok, stats, output_path = _annotate(segmentation)

    assert ok
    assert stats["total_polygons"] == GRID * GRID
    assert stats["matched_polygons"] == GRID * 4
    assert stats["written_polygons"] == GRID * 4
    assert len(json.loads(open(output_path).read())["features"]) == GRID * 4


def test_keep_unassigned_writes_every_polygon_with_a_label(segmentation):
    ok, stats, output_path = _annotate(segmentation, name="keep", keep_unassigned=True)

    assert ok
    assert stats["written_polygons"] == GRID * GRID
    assert stats["label_counts"][DEFAULT_UNASSIGNED_LABEL] == GRID * 2
    labels = {feature_label(f) for f in json.loads(open(output_path).read())["features"]}
    assert labels == set(PHENOTYPES) | {DEFAULT_UNASSIGNED_LABEL}


def test_filter_criteria_restrict_which_cells_are_tagged(segmentation):
    _, stats, _ = _annotate(
        segmentation,
        name="filtered",
        filter_criteria={"phenotype_leiden": ["CAF", "Tumor"], "region": ["core"]},
    )

    assert set(stats["label_counts"]) <= {"CAF", "Tumor"}
    assert stats["written_polygons"] == stats["matched_polygons"] == sum(stats["label_counts"].values())


def test_every_written_label_gets_a_distinct_non_black_color(segmentation):
    _, stats, output_path = _annotate(segmentation, name="colors")

    colors = [f["properties"]["classification"]["colorRGB"] for f in json.loads(open(output_path).read())["features"]]
    assert 0 not in colors, "labels must not fall back to black"
    assert len(set(stats["color_mapping"][p] for p in PHENOTYPES)) == len(PHENOTYPES)


def test_explicit_colors_win_over_the_palette(segmentation):
    _, stats, output_path = _annotate(segmentation, name="custom", color_mapping={"CAF": "#ff0000"})

    assert stats["color_mapping"]["CAF"] == "#ff0000"
    caf = [
        f for f in json.loads(open(output_path).read())["features"]
        if feature_label(f) == "CAF"
    ]
    assert caf and all(f["properties"]["classification"]["colorRGB"] == 0xFF0000 for f in caf)


def test_dataframe_input_and_custom_centroid_columns(segmentation):
    df = pd.read_csv(segmentation["csv"]).rename(columns={"x_centroid": "cx", "y_centroid": "cy"})
    stats: dict = {}

    ok = annotate_geojson_with_phenotypes(
        csv_path=None,
        dataframe=df,
        geojson_path=segmentation["geojson"],
        output_path=str(segmentation["dir"] / "renamed.geojson"),
        phenotype_col="phenotype_leiden",
        x_col="cx",
        y_col="cy",
        generate_png=False,
        stats=stats,
    )

    assert ok
    assert stats["matched_polygons"] == GRID * 4


def test_missing_columns_fail_without_writing_output(segmentation):
    output_path = segmentation["dir"] / "missing.geojson"

    assert not annotate_geojson_with_phenotypes(
        csv_path=segmentation["csv"],
        geojson_path=segmentation["geojson"],
        output_path=str(output_path),
        phenotype_col="not_a_column",
        generate_png=False,
    )
    assert not output_path.exists()


def test_png_is_written_alongside_the_tagged_geojson(segmentation):
    png_path = segmentation["dir"] / "tagged.png"

    ok, _, _ = _annotate(
        segmentation,
        name="with_png",
        generate_png=True,
        png_output_path=str(png_path),
        dpi=60,
        png_title="S1 | phenotype_leiden",
    )

    assert ok
    assert png_path.exists() and png_path.stat().st_size > 0


def test_regenerating_a_png_can_restrict_the_labels_drawn(segmentation):
    _, _, output_path = _annotate(segmentation, name="for_render")
    png_path = segmentation["dir"] / "subset.png"

    assert generate_png_from_geojson(output_path, str(png_path), include_labels=["CAF"], dpi=60)
    assert png_path.stat().st_size > 0
    assert not generate_png_from_geojson(
        output_path, str(segmentation["dir"] / "none.png"), include_labels=["missing"], dpi=60
    )


def test_palette_and_color_mapping_helpers():
    assert len(set(default_palette(60))) == 60
    mapping = build_color_mapping(["b", "a"], existing={"a": "#123456"})
    assert mapping["a"] == "#123456"
    assert mapping["b"].startswith("#") and mapping["b"] != mapping["a"]
    # Sorted assignment keeps colors stable no matter the discovery order.
    assert build_color_mapping(["a", "b"]) == build_color_mapping(["b", "a"])


def test_feature_label_reads_both_classification_shapes():
    assert feature_label({"properties": {"classification": {"name": "CAF"}}}) == "CAF"
    assert feature_label({"properties": {"classification": "CAF"}}) == "CAF"
    assert feature_label({"properties": {}}) == DEFAULT_UNASSIGNED_LABEL


def test_collect_geojson_entries_searches_the_nested_geojson_folder(tmp_path):
    from spacemap_gui.tabs.tab7_phenotyping.tools import _collect_geojson_entries

    (tmp_path / "A_cell_boundaries.geojson").write_text("{}")
    nested = tmp_path / "geojson"
    nested.mkdir()
    (nested / "B_cell_boundaries.geojson").write_text("{}")
    (tmp_path / "ignore.txt").write_text("x")

    entries = _collect_geojson_entries(str(tmp_path), "*_cell_boundaries.geojson")

    assert sorted(e["prefix"] for e in entries) == ["A", "B"]
    assert _collect_geojson_entries("/does/not/exist", "*.geojson") == []


def test_describe_filter_criterion_renders_each_condition_type():
    from spacemap_gui.tabs.tab7_phenotyping.tools import _describe_filter_criterion

    assert _describe_filter_criterion(
        {"column": "area", "type": "numeric", "mode": "between", "value": 1.0, "value_2": 2.0}
    ) == "area in [1, 2]"
    assert _describe_filter_criterion(
        {"column": "area", "type": "numeric", "mode": "gte", "value": 5.0}
    ) == "area ≥ 5"
    assert _describe_filter_criterion(
        {"column": "region", "type": "categorical", "values": ["core", "edge"]}
    ) == "region in (core, edge)"


def test_gui_filters_narrow_the_frame_the_same_way_acc_utils_does(segmentation):
    from spacemap_gui.tabs.tab7_phenotyping.tools import (
        _apply_visual_filters,
        _sanitize_visual_filters,
    )

    df = pd.read_csv(segmentation["csv"])
    criteria = _sanitize_visual_filters(
        [
            {"column": "region", "type": "categorical", "values": ["core"]},
            {"column": "area", "type": "numeric", "mode": "gte", "value": 10.0},
            {"column": "nope", "type": "categorical", "values": ["x"]},
        ],
        list(df.columns),
    )

    assert len(criteria) == 2, "unknown columns are dropped"
    filtered = _apply_visual_filters(df, criteria)
    assert set(filtered["region"]) == {"core"}
    assert filtered["area"].min() >= 10.0


def test_multiselect_selection_is_reset_only_when_its_options_change():
    import streamlit as st

    from spacemap_gui.tabs.tab7_phenotyping.tools import _reset_multiselect_when_options_change

    st.session_state.clear()

    # Rendered before the scan: no options, so the widget stores an empty selection.
    _reset_multiselect_when_options_change("w", [])
    st.session_state["w"] = []

    # The scan brings options in; the stale empty selection must go so that
    # "select all by default" applies.
    _reset_multiselect_when_options_change("w", ["a", "b"])
    assert "w" not in st.session_state

    # A deliberate user choice survives while the options stay the same.
    st.session_state["w"] = ["a"]
    _reset_multiselect_when_options_change("w", ["a", "b"])
    assert st.session_state["w"] == ["a"]


def test_keeping_unassigned_still_writes_a_section_with_no_matching_cells(segmentation):
    """A batch filtered down to nothing must still produce a fully unassigned file."""
    empty = pd.read_csv(segmentation["csv"]).iloc[0:0]
    output_path = segmentation["dir"] / "empty.geojson"
    stats: dict = {}

    ok = annotate_geojson_with_phenotypes(
        csv_path=None,
        dataframe=empty,
        geojson_path=segmentation["geojson"],
        output_path=str(output_path),
        phenotype_col="phenotype_leiden",
        keep_unassigned=True,
        generate_png=False,
        stats=stats,
    )

    assert ok
    assert stats["matched_polygons"] == 0
    assert stats["written_polygons"] == GRID * GRID
    assert stats["label_counts"] == {DEFAULT_UNASSIGNED_LABEL: GRID * GRID}


def test_dropping_unassigned_with_no_matching_cells_fails_loudly(segmentation):
    output_path = segmentation["dir"] / "empty_dropped.geojson"

    assert not annotate_geojson_with_phenotypes(
        csv_path=None,
        dataframe=pd.read_csv(segmentation["csv"]).iloc[0:0],
        geojson_path=segmentation["geojson"],
        output_path=str(output_path),
        phenotype_col="phenotype_leiden",
        keep_unassigned=False,
        generate_png=False,
    )
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Output must stay byte-comparable with the segmentation files QuPath reads
# ---------------------------------------------------------------------------


def test_output_keeps_source_properties_and_adds_no_object_type(segmentation):
    _, _, output_path = _annotate(segmentation, name="props")

    feature = json.loads(open(output_path).read())["features"][0]
    assert feature["properties"]["id"], "source properties must survive"
    assert "objectType" not in feature["properties"], (
        "an injected objectType moves the cells into QuPath's detection hierarchy"
    )
    assert set(feature["properties"]) == {"id", "classification"}


def test_output_is_written_compactly_like_the_source(segmentation):
    _, _, output_path = _annotate(segmentation, name="compact")

    text = Path(output_path).read_text()
    assert "\n" not in text, "pretty-printing inflates a 38 MB section to ~200 MB"
    assert ", " not in text.split('"coordinates"')[1][:200]


def test_pretty_output_is_still_available_on_request(segmentation):
    _, _, output_path = _annotate(segmentation, name="pretty", compact_output=False)

    assert "\n" in Path(output_path).read_text()


# ---------------------------------------------------------------------------
# Missing labels
# ---------------------------------------------------------------------------


def test_missing_label_mask_covers_nulls_and_null_like_text():
    series = pd.Series(["CD8", None, float("nan"), "None", "none", " NA ", "", "Others"])

    mask = missing_label_mask(series)

    assert mask.tolist() == [False, True, True, True, True, True, True, False]


def test_custom_missing_tokens_replace_the_defaults():
    series = pd.Series(["CD8", "None", "Others"])

    mask = missing_label_mask(series, na_values=["Others"])

    # "None" is no longer a token, but a real null would still be missing.
    assert mask.tolist() == [False, False, True]


def test_cells_with_no_label_are_not_tagged_as_the_string_none(segmentation):
    """A sparse manual reference must not paint every cell with a "None" class."""
    df = pd.read_csv(segmentation["csv"])
    # Mimic pyarrow reading SQL nulls out of a mostly-empty reference column.
    df["phenotype_ref"] = [PHENOTYPES[0]] * 3 + ["None"] * (len(df) - 3)
    stats: dict = {}

    annotate_geojson_with_phenotypes(
        csv_path=None,
        dataframe=df,
        geojson_path=segmentation["geojson"],
        output_path=str(segmentation["dir"] / "sparse.geojson"),
        phenotype_col="phenotype_ref",
        generate_png=False,
        stats=stats,
    )

    assert stats["csv_rows"] == 3
    assert stats["csv_rows_missing_label"] == len(df) - 3
    assert stats["written_polygons"] == 3
    assert set(stats["label_counts"]) == {PHENOTYPES[0]}


def test_missing_labels_become_unassigned_when_cells_are_kept(segmentation):
    df = pd.read_csv(segmentation["csv"])
    df["phenotype_ref"] = [PHENOTYPES[0]] * 3 + [None] * (len(df) - 3)
    stats: dict = {}

    annotate_geojson_with_phenotypes(
        csv_path=None,
        dataframe=df,
        geojson_path=segmentation["geojson"],
        output_path=str(segmentation["dir"] / "sparse_kept.geojson"),
        phenotype_col="phenotype_ref",
        keep_unassigned=True,
        generate_png=False,
        stats=stats,
    )

    assert stats["label_counts"][PHENOTYPES[0]] == 3
    assert stats["label_counts"][DEFAULT_UNASSIGNED_LABEL] == GRID * GRID - 3


# ---------------------------------------------------------------------------
# Reference-vs-comparison agreement
# ---------------------------------------------------------------------------


def test_agreement_labels_score_matches_mismatches_and_passthroughs():
    reference = pd.Series(["CD8", "CD8", "Others", None, "CD4", "cd4"])
    comparison = pd.Series(["CD8", "CD4", "CD8", "CD8", None, "CD4"])

    labels = build_agreement_labels(reference, comparison, passthrough_values=["Others"])

    assert labels.tolist()[:3] == ["Match", "Mismatch", "Others"]
    assert pd.isna(labels.iloc[3]), "no reference call means the cell is not scored"
    assert labels.iloc[4] == "Mismatch", "missing comparison counts as a disagreement"
    assert labels.iloc[5] == "Match", "comparison is case-insensitive by default"


def test_agreement_labels_can_be_case_sensitive_and_renamed():
    labels = build_agreement_labels(
        pd.Series(["CD4"]),
        pd.Series(["cd4"]),
        match_label="Same",
        mismatch_label="Different",
        case_sensitive=True,
    )

    assert labels.tolist() == ["Different"]


def test_passthrough_values_win_over_a_coincidental_match():
    labels = build_agreement_labels(
        pd.Series(["Others"]), pd.Series(["Others"]), passthrough_values=["Others"]
    )

    assert labels.tolist() == ["Others"], "an unscored category is never counted as a match"


def test_agreement_labels_tag_a_geojson_end_to_end(segmentation):
    df = pd.read_csv(segmentation["csv"])
    df["reference"] = df["phenotype_leiden"]
    # Half agree, a quarter disagree, a quarter are unscored "Others".
    df.loc[df.index % 4 == 1, "phenotype_leiden"] = "Something Else"
    df.loc[df.index % 4 == 2, "reference"] = "Others"
    df["__pg_agreement__"] = build_agreement_labels(
        df["reference"], df["phenotype_leiden"], passthrough_values=["Others"]
    )
    stats: dict = {}

    annotate_geojson_with_phenotypes(
        csv_path=None,
        dataframe=df,
        geojson_path=segmentation["geojson"],
        output_path=str(segmentation["dir"] / "agreement.geojson"),
        phenotype_col="__pg_agreement__",
        color_mapping={"Match": "#2ca02c", "Mismatch": "#d62728", "Others": "#bfbfbf"},
        generate_png=False,
        stats=stats,
    )

    assert set(stats["label_counts"]) == {"Match", "Mismatch", "Others"}
    assert stats["color_mapping"]["Match"] == "#2ca02c"
    assert stats["color_mapping"]["Mismatch"] == "#d62728"


# ---------------------------------------------------------------------------
# PNG appearance
# ---------------------------------------------------------------------------


def test_png_styling_options_are_accepted_end_to_end(segmentation):
    thin = segmentation["dir"] / "thin.png"
    none = segmentation["dir"] / "no_edge.png"

    _, _, output_path = _annotate(segmentation, name="styled")
    assert generate_png_from_geojson(
        output_path, str(thin), line_width=1.5, edge_color="#000000", alpha=0.5, dpi=60
    )
    # A zero width must mean "no outline", not "thinnest possible".
    assert generate_png_from_geojson(
        output_path, str(none), line_width=0.0, background_color="#000000", dpi=60
    )
    assert thin.stat().st_size > 0 and none.stat().st_size > 0
    assert none.stat().st_size != thin.stat().st_size


def test_gui_parses_the_missing_value_token_list():
    from spacemap_gui.tabs.tab7_phenotyping.tools import _parse_na_tokens

    assert _parse_na_tokens("None, NaN , ,NA") == ["None", "NaN", "NA"]
    assert _parse_na_tokens("") == []


# ---------------------------------------------------------------------------
# Color defaults and the project-wide registry
# ---------------------------------------------------------------------------


def test_known_cell_types_use_the_house_palette():
    """Assert the palette is consulted, not what its colours happen to be.

    DEFAULT_PHENOTYPE_COLORS is a house scheme people tune; pinning specific hex
    values here would make every future tweak look like a regression.
    """
    from spacemap.acc_utils import DEFAULT_PHENOTYPE_COLORS

    mapping = build_color_mapping(["Macrophage", "CD8", "Endothelial"])

    for label in ("Macrophage", "CD8", "Endothelial"):
        assert mapping[label].lower() == DEFAULT_PHENOTYPE_COLORS[label].lower()
    assert len({mapping[label].lower() for label in mapping}) == 3, "distinct cell types, distinct colours"


def test_house_palette_matches_case_insensitively_and_through_aliases():
    from spacemap.acc_utils import DEFAULT_PHENOTYPE_COLORS as HOUSE

    mapping = build_color_mapping(["macrophage", "CAF", "B cell"])

    assert mapping["macrophage"].lower() == HOUSE["Macrophage"].lower(), "lookup ignores case"
    assert mapping["CAF"].lower() == HOUSE["Fibroblast"].lower(), "CAF is an alias of Fibroblast"
    assert mapping["B cell"].lower() == HOUSE["B"].lower(), "B cell is an alias of B"


def test_saved_colors_beat_the_house_palette():
    from spacemap.acc_utils import DEFAULT_PHENOTYPE_COLORS as HOUSE

    mapping = build_color_mapping(["CD8", "Macrophage"], existing={"CD8": "#123456"})

    assert mapping["CD8"] == "#123456"
    assert mapping["Macrophage"].lower() == HOUSE["Macrophage"].lower()


def test_unknown_labels_still_fall_through_to_the_cycling_palette():
    from spacemap.acc_utils import DEFAULT_PHENOTYPE_COLORS as HOUSE

    mapping = build_color_mapping(["Nonesuch", "CD8"])

    assert mapping["CD8"].lower() == HOUSE["CD8"].lower()
    assert mapping["Nonesuch"].startswith("#")
    assert mapping["Nonesuch"].lower() != mapping["CD8"].lower()


def test_house_palette_can_be_switched_off():
    from spacemap.acc_utils import DEFAULT_PHENOTYPE_COLORS as HOUSE

    mapping = build_color_mapping(["CD8"], use_known_colors=False)

    assert mapping["CD8"].lower() != HOUSE["CD8"].lower()


def test_color_registry_migrates_legacy_per_column_maps():
    from spacemap_gui.tabs.tab7_phenotyping.tools import _load_pg_color_registry

    registry = _load_pg_color_registry(
        {
            "tools_pg_label_colors": {"CD8": "#111111"},
            "tools_pg_color_maps": {
                "phenotype_leiden": {"CD8": "#999999", "CAF": "#222222"},
                "phenotype_som": {"Treg": "#333333"},
            },
        }
    )

    assert registry["CD8"] == "#111111", "the current registry wins over a legacy entry"
    assert registry["CAF"] == "#222222"
    assert registry["Treg"] == "#333333"


def test_registry_lookup_is_case_insensitive():
    from spacemap_gui.tabs.tab7_phenotyping.tools import _registry_colors_for

    resolved = _registry_colors_for({"CD8": "#111111"}, ["cd8", " CD8 ", "CD4"])

    assert resolved["cd8"] == "#111111"
    assert resolved[" CD8 "] == "#111111"
    assert "CD4" not in resolved


def test_color_widget_key_is_shared_across_columns_but_unique_per_label():
    from spacemap_gui.tabs.tab7_phenotyping.tools import _color_widget_key

    # The old key mixed in the column, so switching the comparison column reset
    # every picker. One label is now one widget, wherever it is shown.
    assert _color_widget_key("CD8") == _color_widget_key("CD8")
    assert _color_widget_key("T cell") != _color_widget_key("T-cell")
    assert " " not in _color_widget_key("T cell / NK")
    assert not _color_widget_key("CD8").startswith("tools_"), (
        "picker keys must stay out of the auto-persisted params"
    )


def test_remembering_colors_updates_the_params_payload():
    from spacemap_gui.tabs.tab7_phenotyping.tools import (
        _PG_COLOR_REGISTRY_KEY,
        _remember_pg_colors,
    )

    params: dict = {}
    registry: dict = {}
    _remember_pg_colors(params, registry, {"CD8": "#111111", "Blank": ""})

    assert params[_PG_COLOR_REGISTRY_KEY] == {"CD8": "#111111"}


# ---------------------------------------------------------------------------
# Agreement summary
# ---------------------------------------------------------------------------


def test_agreement_summary_excludes_ignored_cells_from_the_percentages():
    from spacemap_gui.tabs.tab7_phenotyping.tools import _agreement_summary

    results = [
        {"batch_id": "S1", "non_assigned": 40, "label_counts": {"Match": 30, "Mismatch": 10, "Others": 20}},
        {"batch_id": "S2", "non_assigned": 5, "label_counts": {"Match": 10, "Mismatch": 10, "Others": 5}},
    ]
    meta = {"match_label": "Match", "mismatch_label": "Mismatch", "passthrough": ["Others"]}

    summary = _agreement_summary(results, meta)

    first = summary.iloc[0]
    assert first["cells_scored"] == 40 and first["ignored"] == 20
    assert first["cells_tagged"] == 60
    assert first["% Match"] == 75.0, "the 20 ignored cells must not dilute the rate"

    total = summary.iloc[-1]
    assert total["batch_id"] == "TOTAL"
    assert total["cells_scored"] == 60 and total["ignored"] == 25 and total["not_scored"] == 45
    assert total["% Match"] == 66.67
    assert round(total["% Match"] + total["% Mismatch"], 2) == 100.0


def test_agreement_summary_follows_renamed_labels():
    from spacemap_gui.tabs.tab7_phenotyping.tools import _agreement_summary

    summary = _agreement_summary(
        [{"batch_id": "S1", "non_assigned": 0, "label_counts": {"Same": 3, "Missmatch": 1}}],
        {"match_label": "Same", "mismatch_label": "Missmatch", "passthrough": []},
    )

    assert list(summary.columns)[3:5] == ["Same", "Missmatch"]
    assert summary.iloc[0]["% Same"] == 75.0


def test_agreement_summary_handles_a_section_with_nothing_scored():
    from spacemap_gui.tabs.tab7_phenotyping.tools import _agreement_summary

    summary = _agreement_summary(
        [{"batch_id": "S1", "non_assigned": 10, "label_counts": {"Others": 4}}],
        {"match_label": "Match", "mismatch_label": "Mismatch", "passthrough": ["Others"]},
    )

    assert summary.iloc[0]["% Match"] == 0.0, "no division by zero when nothing is scorable"
    assert summary.iloc[0]["ignored"] == 4
    assert _agreement_summary([], {}).empty
