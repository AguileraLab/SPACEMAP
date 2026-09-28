"""Tab 5's segmentation region and spatial columns for one image, from saved settings (as Studio runs it)."""

import json
from types import SimpleNamespace

from spacemap_gui.tabs import segmentation_spatial_ui as spatial


def _annotation(path, classes):
    features = [
        {"type": "Feature", "properties": {"classification": {"name": name}},
         "geometry": {"type": "Polygon", "coordinates": [[[i * 10, 0], [i * 10 + 9, 0], [i * 10 + 9, 9], [i * 10, 9], [i * 10, 0]]]}}
        for i, name in enumerate(classes)
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def _state(folder, **extra):
    return {spatial.GEOJSON_DIR_KEY: str(folder), spatial.MASK_CLASSES_KEY: ["tissue"], spatial.SPATIAL_ENABLED_KEY: True, **extra}


def test_whole_image_or_inside_annotation_classes(tmp_path):
    _annotation(tmp_path / "P1_roi_1_zones.geojson", ["tissue", "Tumor"])
    config = SimpleNamespace()
    notes = spatial.apply_region_and_tags(config, "P1_roi_1.ome.tiff", spatial.MASK_SOURCE_NONE, state=_state(tmp_path))
    assert config.use_input_geojson is False and notes == []
    notes = spatial.apply_region_and_tags(config, "P1_roi_1.ome.tiff", spatial.MASK_SOURCE_ANNOTATION, state=_state(tmp_path))
    assert config.use_input_geojson and config.mask_class_names == ["tissue"]
    assert config.input_geojson_path.endswith("P1_roi_1_zones.geojson") and "inside 'tissue'" in notes[0]


def test_an_annotation_chosen_by_hand_for_a_mismatched_name(tmp_path):
    _annotation(tmp_path / "scan 7 final.geojson", ["tissue"])
    config = SimpleNamespace()
    state = _state(tmp_path)
    spatial.apply_region_and_tags(config, "slide7.ome.tiff", spatial.MASK_SOURCE_ANNOTATION, state=state)
    assert config.use_input_geojson is False  # no file matches the name
    state[spatial.GEOJSON_MATCHES_KEY] = {"slide7.ome.tiff": "scan 7 final.geojson"}
    spatial.apply_region_and_tags(config, "slide7.ome.tiff", spatial.MASK_SOURCE_ANNOTATION, state=state)
    assert config.use_input_geojson and config.input_geojson_path.endswith("scan 7 final.geojson")
    state[spatial.GEOJSON_MATCHES_KEY] = {"slide7.ome.tiff": "../outside.geojson"}  # only files in the folder
    spatial.apply_region_and_tags(config, "slide7.ome.tiff", spatial.MASK_SOURCE_ANNOTATION, state=state)
    assert config.use_input_geojson is False


def test_saved_rules_add_spatial_columns(tmp_path):
    _annotation(tmp_path / "P1_roi_1_zones.geojson", ["tissue", "Tumor"])
    rules = [{"Annotation Label": "Tumor", "Binary (0/1)": True, "Distance": True}, {"Annotation Label": "Gone"}]
    config = SimpleNamespace()
    notes = spatial.apply_region_and_tags(config, "P1_roi_1.ome.tiff", spatial.MASK_SOURCE_NONE,
                                          state=_state(tmp_path, **{spatial.SPATIAL_RULES_KEY: rules}))
    assert [rule["Annotation Label"] for rule in config.spatial_rules] == ["Tumor"] and config.spatial_rules[0]["Group by Tissue ROI"]
    assert config.spatial_tissue_label == "tissue" and notes == ["Spatial annotation: 1 annotation rule(s) from P1_roi_1_zones.geojson"]
