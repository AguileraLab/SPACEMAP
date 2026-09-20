from __future__ import annotations

import pandas as pd
from PIL import Image

from spacemap_gui.tabs.tab9_visualization import cell_visualization
from spacemap_gui.tabs.tab9_visualization import overlay_core


def _rectangle_feature(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    name: str = "Tumor",
) -> dict:
    return {
        "type": "Feature",
        "properties": {"classification": {"name": name, "color": [200, 0, 0]}},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]],
        },
    }


def test_crop_base_to_fullres_box_returns_matching_effective_coordinates() -> None:
    image = Image.new("RGB", (100, 50), "black")

    cropped, effective = overlay_core.crop_base_to_fullres_box(
        image,
        full_shape=(100, 200),
        crop_box=(41, 21, 159, 79),
    )

    # The preview is at half resolution.  floor/ceil retains the requested box,
    # and the effective level-0 box describes those exact integer preview pixels.
    assert cropped.size == (60, 30)
    assert effective == (40.0, 20.0, 160.0, 80.0)


def test_legacy_all_black_category_palette_is_seeded_from_house_colors() -> None:
    colors = cell_visualization._initial_category_colors(
        ["B", "CD4", "Tumor"],
        {"B": "#000000", "CD4": "#000000", "Tumor": "#000000"},
        migrate_legacy_all_black=True,
    )

    assert colors["B"] != "#000000"
    assert colors["CD4"] != "#000000"
    assert colors["Tumor"] != "#000000"


def test_individual_black_category_choice_is_preserved() -> None:
    colors = cell_visualization._initial_category_colors(
        ["B", "CD4"],
        {"B": "#000000", "CD4": "#123456"},
        migrate_legacy_all_black=True,
    )

    assert colors == {"B": "#000000", "CD4": "#123456"}


def test_comparison_columns_are_included_in_bottom_caption() -> None:
    text = cell_visualization._title_text(
        record={"roi": "reg018_B", "has_image": True, "has_cells": True, "has_tissue": True},
        parts=["ROI name", "Cell count"],
        custom="",
        markers=[],
        info={
            "cells": 2743,
            "reference_column": "phenotype_ref1",
            "compared_column": "phenotype_ml_consensus",
        },
    )

    assert text == (
        "reg018_B   |   2,743 cells   |   Reference: phenotype_ref1"
        "   |   Compared: phenotype_ml_consensus"
    )


def test_multiple_comparison_columns_create_ordered_render_jobs_and_filenames() -> None:
    jobs = cell_visualization._comparison_columns_to_render(
        "Match vs reference",
        ["phenotype_som", "phenotype_leiden", "phenotype_som", ""],
    )

    assert jobs == ["phenotype_som", "phenotype_leiden"]
    assert cell_visualization._comparison_filename_suffix(
        "phenotype_ref1", "phenotype_som"
    ) == "_phenotype_som_vs_phenotype_ref1"
    assert cell_visualization._comparison_columns_to_render("Fixed", jobs) == [""]


def test_cell_filters_combine_rules_and_always_drop_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "phenotype": ["CD4", "CD4", "CD8", None, ""],
            "score": [0.2, 0.8, 0.9, 0.7, 0.6],
        }
    )

    filtered = cell_visualization.apply_cell_filters(
        frame,
        [
            {"column": "phenotype", "type": "Categorical", "values": ["CD4"]},
            {
                "column": "score",
                "type": "Continuous",
                "operator": "Above (>=)",
                "threshold": 0.5,
            },
        ],
    )

    assert filtered is not None
    assert filtered.index.tolist() == [1]


def test_continuous_cell_filter_supports_between_and_below() -> None:
    frame = pd.DataFrame({"score": [None, 0.1, 0.5, 0.8, 1.0]})

    between = cell_visualization.apply_cell_filters(
        frame,
        [
            {
                "column": "score",
                "type": "Continuous",
                "operator": "Between",
                "minimum": 0.2,
                "maximum": 0.8,
            }
        ],
    )
    below = cell_visualization.apply_cell_filters(
        frame,
        [
            {
                "column": "score",
                "type": "Continuous",
                "operator": "Below (<=)",
                "threshold": 0.5,
            }
        ],
    )

    assert between is not None and between.index.tolist() == [2, 3]
    assert below is not None and below.index.tolist() == [1, 2]


def test_filtered_segmentation_features_are_matched_by_normalized_cell_label() -> None:
    features = [
        {"properties": {"label": 1}, "geometry": _rectangle_feature(0, 0, 1, 1)["geometry"]},
        {"properties": {"label": "2"}, "geometry": _rectangle_feature(2, 0, 3, 1)["geometry"]},
    ]
    retained, count = cell_visualization._filter_features_to_frame_labels(
        features,
        pd.DataFrame({"label": [1.0]}),
        frame_label_column="label",
    )

    assert count == 1
    assert retained[0]["properties"]["label"] == 1


def test_auto_zoom_bounds_contain_all_filtered_cells_with_padding() -> None:
    cells = [
        _rectangle_feature(10, 20, 20, 30, "cell-a"),
        _rectangle_feature(30, 40, 50, 60, "cell-b"),
    ]

    bounds = cell_visualization._padded_feature_bounds(cells, 10.0, full_shape=(100, 100))

    assert bounds == (6.0, 16.0, 54.0, 64.0)


def test_cells_are_clipped_to_the_tissue_containing_their_centroid() -> None:
    tissues = [
        _rectangle_feature(0, 0, 5, 10, "Tumor"),
        _rectangle_feature(5, 0, 10, 10, "Others"),
    ]
    cells = [
        _rectangle_feature(4, 3, 5.5, 5, "cell-left"),
        _rectangle_feature(4.5, 6, 7, 8, "cell-right"),
        _rectangle_feature(11, 1, 12, 2, "cell-outside"),
    ]

    clipped, stats = overlay_core.clip_cells_to_containing_tissue(cells, tissues)

    assert len(clipped) == 2
    assert stats == {"input": 3, "assigned": 2, "geometry_clipped": 2, "outside": 1}
    assert overlay_core.geometry_bounds(clipped[0]["geometry"]) == (4.0, 3.0, 5.0, 5.0)
    assert clipped[0]["properties"]["tissue_annotation"] == "Tumor"
    assert overlay_core.geometry_bounds(clipped[1]["geometry"]) == (5.0, 6.0, 7.0, 8.0)
    assert clipped[1]["properties"]["tissue_annotation"] == "Others"


def test_build_figure_crops_base_before_mapping_tissue_and_cells(monkeypatch) -> None:
    feature = _rectangle_feature(40, 20, 160, 80)

    monkeypatch.setattr(
        overlay_core,
        "load_base_image",
        lambda *_args, **_kwargs: (Image.new("RGB", (200, 100), "black"), (100, 200), 0),
    )
    monkeypatch.setattr(overlay_core, "features_for_classes", lambda *_args, **_kwargs: [feature])

    image, info = cell_visualization._build_figure(
        record={
            "roi": "example",
            "image": "example.ome.tiff",
            "tissue": "example.geojson",
            "cells": "",
            "has_image": True,
            "has_tissue": True,
            "has_cells": False,
        },
        loaded_df=pd.DataFrame(),
        viz_settings={},
        base_source="ROI image",
        markers=[],
        marker_colors={},
        level_choice="auto",
        max_size=2000,
        contrast=(1.0, 99.5),
        background="#000000",
        tissue_classes=["Tumor"],
        tissue_use_embedded=True,
        tissue_color="#C80000",
        tissue_alpha=55,
        tissue_outline_w=2,
        crop_to_tissue=True,
        mask_outside=False,
        cell_source="None",
        cell_alpha=170,
        cell_colour_mode="Fixed",
        cell_fixed="#000000",
        radius=30,
        layer_order=("tissue", "cells"),
        show_legend=False,
        show_title=False,
        title_parts=[],
        custom_title="",
    )

    assert image.size == (120, 60)
    assert info["size"] == "120x60"
    assert info["crop"] == (40, 20, 160, 80)


def test_build_figure_filters_segmentation_polygons_and_zooms_to_survivors(monkeypatch) -> None:
    retained_cell = _rectangle_feature(10, 10, 20, 20, "cell-retained")
    retained_cell["properties"] = {"label": 1}
    excluded_cell = _rectangle_feature(70, 70, 90, 90, "cell-excluded")
    excluded_cell["properties"] = {"label": 2}

    monkeypatch.setattr(
        overlay_core,
        "load_base_image",
        lambda *_args, **_kwargs: (Image.new("RGB", (100, 100), "black"), (100, 100), 0),
    )
    monkeypatch.setattr(
        overlay_core,
        "load_geojson",
        lambda *_args, **_kwargs: {
            "type": "FeatureCollection",
            "features": [retained_cell, excluded_cell],
        },
    )

    image, info = cell_visualization._build_figure(
        record={
            "roi": "example",
            "image": "example.ome.tiff",
            "tissue": "",
            "cells": "example_cells.geojson",
            "has_image": True,
            "has_tissue": False,
            "has_cells": True,
        },
        loaded_df=pd.DataFrame(
            {
                "batch_id": ["example", "example"],
                "label": [1.0, 2.0],
                "phenotype": ["CD4", "CD8"],
            }
        ),
        viz_settings={"batch_col": "batch_id"},
        base_source="ROI image",
        markers=[],
        marker_colors={},
        level_choice="auto",
        max_size=2000,
        contrast=(1.0, 99.5),
        background="#000000",
        tissue_classes=[],
        tissue_use_embedded=True,
        tissue_color="#C80000",
        tissue_alpha=55,
        tissue_outline_w=2,
        crop_to_tissue=False,
        mask_outside=False,
        cell_filter_rules=[
            {"column": "phenotype", "type": "Categorical", "values": ["CD4"]}
        ],
        auto_zoom_filtered_cells=True,
        zoom_padding=0.0,
        cell_source="Segmentation polygons",
        cell_alpha=170,
        cell_colour_mode="Fixed",
        cell_fixed="#FFFFFF",
        radius=30,
        layer_order=("tissue", "cells"),
        show_legend=False,
        show_title=False,
        title_parts=[],
        custom_title="",
    )

    assert info["cell_filter"] == "1/2 retained"
    assert info["segmentation_filter"] == "1 polygons retained"
    assert info["cells"] == 1
    assert info["crop_source"] == "filtered cells"
    assert info["crop"] == (10, 10, 20, 20)
    assert image.size == (10, 10)
