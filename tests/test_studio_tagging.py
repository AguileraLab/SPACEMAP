"""Studio's Cell tagging step runs Tab 5's spatial cell tagging from its saved settings;
Segmentation's "where to segment" is Tab 5's mask, applied as classic Tab 5 applies it."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from spacemap_gui.studio import segmentation, tagging
from spacemap_gui.studio.settings import read_config
from spacemap_gui.tabs import cell_tagging_ui as tab5


def _polygon(name, x0, y0, x1, y1):
    return {"type": "Feature", "properties": {"classification": {"name": name}},
            "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


def _project(root):
    """Two ROIs: P1's annotation matches by name, P2's has an odd name. Tumor is the left half of the tissue."""
    tables = root / "outputs/segmentation/cellpose/features"
    notes = root / "outputs/processed_annotations/stage1b_compartments_zones"
    tables.mkdir(parents=True)
    notes.mkdir(parents=True)
    for roi in ("P1_roi_1", "P2_roi_1"):
        pd.DataFrame({"label": range(1, 7), "x_centroid": [10, 30, 45, 60, 80, 95.0], "y_centroid": [50.0] * 6,
                      "DAPI": 1.0}).to_csv(tables / f"{roi}_cell_features.csv", index=False)
    annotation = {"type": "FeatureCollection", "features": [_polygon("tissue", 0, 0, 100, 100), _polygon("Tumor", 0, 0, 50, 100),
                                                               _polygon("Stroma", 50, 0, 100, 100)]}
    (notes / "P1_roi_1_zones.geojson").write_text(json.dumps(annotation))
    (notes / "slide two final.geojson").write_text(json.dumps(annotation))
    (root / f"{root.name}.json").write_text(json.dumps({"project_name": root.name, "directory_mappings": {}, "tab_configurations": {}}))
    return root


def _save(root, changes):
    from spacemap_gui.studio.settings import revision

    current = tagging.values(root, read_config(root))
    return tagging.save(root, changes, revision(current))


def test_tables_and_annotations_are_found_and_mismatches_fixed_by_hand(tmp_path):
    root = _project(tmp_path)
    state = tagging.values(root, read_config(root))
    assert state[tab5.MODE_KEY] == tab5.MODE_PER_IMAGE and state[tab5.OUTPUT_SUFFIX_KEY] == "_tagged"
    found = tagging.sources(root, state)
    assert [(row["source"], row["annotation"]) for row in found["rows"]] == [
        ("P1_roi_1_cell_features.csv", "P1_roi_1_zones.geojson"), ("P2_roi_1_cell_features.csv", None)]
    assert sorted(found["classes"]) == ["Stroma", "Tumor", "tissue"] and found["coordinates"] == {"x": "x_centroid", "y": "y_centroid"}
    state = _save(root, {tab5.GEOJSON_MATCHES_KEY: {"P2_roi_1_cell_features.csv": "slide two final.geojson"}})
    assert tagging.sources(root, state)["rows"][1]["annotation"] == "slide two final.geojson"
    saved = read_config(root)["tab_configurations"]["tab5_segmentation"]  # where classic Tab 5 keeps it
    assert saved[tab5.GEOJSON_MATCHES_KEY] == {"P2_roi_1_cell_features.csv": "slide two final.geojson"}


def test_settings_are_checked(tmp_path):
    root = _project(tmp_path)
    for wrong in ({tab5.MODE_KEY: "both"}, {tab5.OUTPUT_SUFFIX_KEY: ""}, {tab5.BED_LABELS_KEY: ["Tumor"]},
                  {tab5.GEOJSON_MATCHES_KEY: {"a.csv": "../b.geojson"}},
                  {tab5.CSV_DIR_KEY: "../up"}, {tab5.MIN_BED_CELLS_KEY: 0}, {"colour": "x"},
                  {"seg_tag_cpu_workers": -1}, {"seg_tag_cpu_workers": True}, {"seg_tag_cpu_workers": 257}):
        with pytest.raises(ValueError):
            tagging.check(wrong)
    with pytest.raises(ValueError, match="another window"):
        tagging.save(root, {tab5.EDGE_ENABLED_KEY: True}, "stale")
    with pytest.raises(ValueError, match="not a folder"):
        _save(root, {tab5.CSV_DIR_KEY: str(tmp_path / "elsewhere")})  # a full path is fine, if it is a folder
    outside = tmp_path / "shared_tables"
    outside.mkdir()
    state = _save(root, {tab5.CSV_DIR_KEY: str(outside)})
    assert state[tab5.CSV_DIR_KEY] == str(outside.resolve())
    state = _save(root, {tab5.OUTPUT_DIR_KEY: "outputs/tagged"})
    assert state[tab5.OUTPUT_DIR_KEY] == str(root / "outputs/tagged")
    assert read_config(root)["tab_configurations"]["tab5_segmentation"][tab5.OUTPUT_DIR_KEY] == "${PROJECT_ROOT}/outputs/tagged"


def _rules(state, root):
    found = tagging.sources(root, state)
    rules = tagging.rules(found, state)
    for rule in rules:
        rule.update({"Binary (0/1)": True, "Distance": rule["Annotation Label"] == "Tumor"})
    return rules


def test_each_table_is_tagged_with_its_own_annotation(tmp_path):
    root = _project(tmp_path)
    state = _save(root, {tab5.GEOJSON_MATCHES_KEY: {"P2_roi_1_cell_features.csv": "slide two final.geojson"}})
    state = _save(root, {tab5.RULES_KEY: _rules(state, root), tab5.BED_LABELS_KEY: ["Tumor", "Stroma"], tab5.EDGE_ENABLED_KEY: True,
                         tab5.EDGE_BUFFER_KEY: 10.0, tab5.MIN_BED_CELLS_KEY: 1})
    assert "Tumor" in tagging.planned_columns(state) and "Tumor_distance" in tagging.planned_columns(state)
    found = tagging.sources(root, state)
    events = []
    result = tagging.run({"state": state, "found": found, "tables": []}, tmp_path / "job",
                         lambda message, progress=None: events.append((message, progress)))
    assert result["count"] == 2 and result["failed"] == 0
    messages = "\n".join(message for message, _ in events)
    for stage in ("Loading cell table", "Loaded 6 cells", "Preparing annotation", "Assigning beds", "Calculating edge cells", "Writing staged table"):
        assert stage in messages
    fractions = [fraction for _, fraction in events if fraction is not None]
    assert fractions == sorted(fractions) and fractions[-1] == .99
    tagged = pd.read_csv(root / "outputs/segmentation/tagged_features/P2_roi_1_cell_features_tagged.csv")
    assert tagged["Tumor"].tolist() == [1, 1, 1, 0, 0, 0] and tagged["Stroma"].tolist() == [0, 0, 0, 1, 1, 1]
    assert (tagged["Tumor_distance"] < 0).tolist() == [True, True, True, False, False, False]
    assert tagged["edge_cell"].tolist() == [1, 0, 0, 0, 0, 1]  # within 10 px of the tissue's edge
    assert tagged["DAPI"].tolist() == [1.0] * 6  # the measurements are kept
    assert tagging.outputs(state, found) == {name: name.replace(".csv", "_tagged.csv") for name in found["tables"]}
    original = pd.read_csv(root / "outputs/segmentation/cellpose/features/P2_roi_1_cell_features.csv")
    assert "Tumor" not in original  # the original table is never changed


def test_a_combined_table_is_tagged_roi_by_roi(tmp_path):
    root = _project(tmp_path)
    tables = root / "outputs/segmentation/cellpose/features"
    joined = pd.concat([pd.read_csv(tables / f"{roi}_cell_features.csv").assign(batch_id=roi) for roi in ("P1_roi_1", "P2_roi_1")])
    (root / "outputs/combined").mkdir(parents=True)
    joined.to_csv(root / "outputs/combined/all_cells.csv", index=False)
    state = _save(root, {tab5.MODE_KEY: tab5.MODE_MERGED, tab5.MERGED_DIR_KEY: "outputs/combined", tab5.MERGED_FILE_KEY: "all_cells.csv",
                         tab5.GEOJSON_MATCHES_KEY: {"P2_roi_1": "slide two final.geojson"}})
    found = tagging.sources(root, state)
    assert found["batch_column"] == "batch_id" and [row["source"] for row in found["rows"]] == ["P1_roi_1", "P2_roi_1"]
    assert all(row["annotation"] for row in found["rows"])
    state = _save(root, {tab5.RULES_KEY: _rules(state, root)})
    result = tagging.run({"state": state, "found": found, "tables": []}, tmp_path / "job", lambda *args: None)
    tagged = pd.read_csv(root / "outputs/segmentation/tagged_features/all_cells_tagged.csv")
    assert result["count"] == 1 and len(tagged) == 12 and tagged["Tumor"].tolist() == [1, 1, 1, 0, 0, 0] * 2


def test_where_to_segment_is_tab5s_mask(tmp_path):
    root = _project(tmp_path)
    settings = segmentation.values(root, read_config(root))
    assert settings["tab4_mask_source"] == "No mask (whole image)" and settings["tab4_annot_geojson_dir"].endswith("stage1b_compartments_zones")
    from spacemap_gui.tabs import segmentation_spatial_ui as spatial

    assert segmentation.DEFAULTS_SOURCES == spatial.MASK_SOURCES
    from spacemap_gui.studio.settings import revision

    settings = segmentation.save(root, {"tab4_mask_source": "Annotation GeoJSON (Annotation tab)", "tab4_mask_classes": ["tissue"],
                                        "tab4_annot_geojson_matches": {"P2_roi_1.ome.tiff": "slide two final.geojson"}}, revision(settings))
    found = segmentation.regions(settings, ["P1_roi_1.ome.tiff", "P2_roi_1.ome.tiff", "P3_roi_1.ome.tiff"])
    assert [row["annotation"] for row in found["rows"]] == ["P1_roi_1_zones.geojson", "slide two final.geojson", None]
    config = SimpleNamespace()  # classic Tab 5 applies its own mask choice
    notes = spatial.apply_region_and_tags(config, "P2_roi_1.ome.tiff", settings["tab4_mask_source"], settings["tab4_seg_mask_save_dir"],
                                          state=segmentation.state_for(settings))
    assert config.use_input_geojson and config.mask_class_names == ["tissue"] and Path(config.input_geojson_path).name == "slide two final.geojson"
    assert "inside 'tissue'" in notes[0]
    studio = SimpleNamespace()  # Studio segments the whole image unless an image's area is one of its annotations
    segmentation.region(studio, "P2_roi_1.ome.tiff", settings)
    assert not studio.use_input_geojson
    for wrong in ({"tab4_mask_source": "everywhere"}, {"tab4_annot_geojson_dir": "/abs"}, {"tab4_annot_geojson_matches": {"a": "x/y"}}):
        with pytest.raises(ValueError):
            segmentation.check(wrong)
    settings = segmentation.save(root, {"tab4_annot_geojson_dir": "outputs/processed_annotations/stage1b_compartments_zones"}, revision(settings))
    assert read_config(root)["directory_mappings"]["tab4_annot_geojson_dir"] == "${PROJECT_ROOT}/outputs/processed_annotations/stage1b_compartments_zones"


def test_a_preview_tags_the_open_roi_in_memory_and_colours_its_cells(tmp_path):
    from spacemap_gui.viewer import api

    root = _project(tmp_path)
    state = _save(root, {tab5.GEOJSON_MATCHES_KEY: {"P2_roi_1_cell_features.csv": "slide two final.geojson"}})
    state = _save(root, {tab5.RULES_KEY: _rules(state, root), tab5.BED_LABELS_KEY: ["Tumor", "Stroma"], tab5.MIN_BED_CELLS_KEY: 1})
    before = sorted(path.name for path in root.rglob("*") if path.is_file())
    image = str(root / "outputs/rois/P2_roi_1.ome.tiff")
    found = tagging.preview(root, state, image, "P2_roi_1")
    assert found["table"] == "P2_roi_1_cell_features.csv" and found["annotation"] == "slide two final.geojson" and found["cells"] == 6
    assert {"Tumor", "Tumor_distance", "tissue_roi"} <= set(found["columns"])
    assert found["summary"]["Tumor"] == {"kind": "categorical", "blank": 0, "counts": [["0", 3], ["1", 3]]} or found["summary"]["Tumor"]["counts"]
    assert sorted(path.name for path in root.rglob("*") if path.is_file()) == before  # nothing written
    layer = {"image": image, "roi": "P2_roi_1", "tables": {}}
    lut, legend = api._cell_lut(layer, "", tagging.PREFIX + "Tumor_distance", frozenset(), "viridis")
    assert lut is not None and legend["kind"] in {"numeric", "categorical"}  # six cells: few distinct values
    assert api._cell_lut(layer, "", tagging.PREFIX + "missing", frozenset(), "viridis") == (None, None)
    tagging.clear_preview(image, "P2_roi_1")
    assert api._cell_lut(layer, "", tagging.PREFIX + "Tumor_distance", frozenset(), "viridis") == (None, None)
    with pytest.raises(ValueError, match="No cell table"):
        tagging.preview(root, state, str(root / "outputs/rois/P9_roi_1.ome.tiff"), "P9_roi_1")


def test_exports_names_units_and_the_bed_column(tmp_path):
    """The Annotation step's export is each table's default annotation; columns take the user's names; distances
    become µm and sizes µm² with the pixel size; one bed column names the annotation each cell was given to."""
    root = _project(tmp_path)
    config = json.loads((root / f"{root.name}.json").read_text())
    config["mpp_codex"] = 0.5
    (root / f"{root.name}.json").write_text(json.dumps(config))
    export = root / "outputs/processed_annotations/stage0_tissue_annotation/studio_export/P2_roi_1_annotations.geojson"
    export.parent.mkdir(parents=True)
    export.write_text(json.dumps({"type": "FeatureCollection", "features": [_polygon("tissue", 0, 0, 100, 100), _polygon("Tumor", 0, 0, 50, 100),
                                                                            _polygon("Stroma", 50, 0, 100, 100)]}))
    state = tagging.values(root, read_config(root))
    assert state[tab5.CLEAN_PATH_KEY] is True and state["seg_tag_units"] == "um"  # Studio's starting choices
    rows = {row["source"]: row for row in tagging.sources(root, state)["rows"]}
    assert rows["P2_roi_1_cell_features.csv"]["exported"] and rows["P2_roi_1_cell_features.csv"]["annotation"] == export.name
    assert not rows["P1_roi_1_cell_features.csv"]["exported"]  # no export: matched by name, as before
    rules = _rules(state, root)
    for rule in rules:
        if rule["Annotation Label"] == "tissue":
            rule.update({"ROI ID": True, "ROI Size": True})
    state = _save(root, {tab5.RULES_KEY: rules, tab5.BED_LABELS_KEY: ["Stroma", "Tumor"], tab5.MIN_BED_CELLS_KEY: 1,
                         "seg_tag_column_names": {"tissue|roi": "tissue_piece", "Tumor|distance": "dist_tumor"}})
    planned = tagging.planned_columns(state, tagging.rules(tagging.sources(root, state), state))
    assert {"tissue_piece", "tissue_roi_size", "dist_tumor", "bed", "Tumor_bed"} <= set(planned)
    with pytest.raises(ValueError):
        tagging.check({"seg_tag_column_names": {"Tumor|distance": "1 bad name"}})
    found = tagging.sources(root, state)
    result = tagging.run({"state": state, "found": found, "tables": ["P2_roi_1_cell_features.csv"], "root": str(root),
                          "exports": tagging._exports_and_images(root)[0]}, None, lambda *a: None)
    assert result["count"] == 1 and result["records"][0]["Status"] == "ok", result
    tagged = pd.read_csv(result["outputs"][0])
    assert tagged["tissue_piece"].eq(1).all() and tagged["tissue_roi_size"].iloc[0] == pytest.approx(100 * 100 * 0.25)  # µm²
    assert tagged.loc[tagged["x_centroid"] == 60, "dist_tumor"].iloc[0] == pytest.approx(10 * 0.5)  # 10 px outside = 5 µm
    assert list(tagged["bed"]) == ["Tumor", "Tumor", "Tumor", "Stroma", "Stroma", "Stroma"]


def test_parquet_tables_per_image_are_listed_and_tagged_as_parquet(tmp_path):
    root = _project(tmp_path)
    tables = root / "outputs/segmentation/cellpose/features"
    for csv in tables.glob("*.csv"):
        pd.read_csv(csv).to_parquet(csv.with_suffix(".parquet"))
        csv.unlink()
    state = _save(root, {tab5.GEOJSON_MATCHES_KEY: {"P2_roi_1_cell_features.parquet": "slide two final.geojson"}})
    found = tagging.sources(root, state)
    assert [row["source"] for row in found["rows"]] == ["P1_roi_1_cell_features.parquet", "P2_roi_1_cell_features.parquet"]
    rules = _rules(state, root)
    state = _save(root, {tab5.RULES_KEY: rules})
    found = tagging.sources(root, state)
    result = tagging.run({"state": state, "found": found, "tables": [], "root": str(root), "exports": {}}, root / "job", lambda *a: None)
    assert result["count"] == 2 and all(path.endswith(".parquet") for path in result["outputs"])


@pytest.mark.parametrize("error_number, message", [
    (122, "Storage quota exceeded"),
    (28, "filesystem is full"),
    (13, "not writable"),
])
def test_failed_settings_write_is_reported_and_keeps_previous_config(tmp_path, monkeypatch, error_number, message):
    from spacemap_gui import persistence

    root = _project(tmp_path)
    path = root / f"{root.name}.json"
    previous = path.read_bytes()
    monkeypatch.setattr(persistence, "attach_portable_state", lambda root, config: config)
    monkeypatch.setattr(persistence, "create_restore_point", lambda *args, **kwargs: None)

    def cannot_create_temp(*args, **kwargs):
        raise OSError(error_number, "simulated write failure")

    monkeypatch.setattr(persistence.tempfile, "mkstemp", cannot_create_temp)
    with pytest.raises(ValueError, match=message):
        _save(root, {tab5.OUTPUT_DIR_KEY: "outputs/new_tagging_folder"})
    assert path.read_bytes() == previous
    assert tagging.values(root, read_config(root))[tab5.OUTPUT_DIR_KEY] != str(root / "outputs/new_tagging_folder")
    # Older autosave callers retain their log-only behavior.
    persistence.save_project_config(str(root), read_config(root))
    assert path.read_bytes() == previous


def test_failed_image_is_explained_in_log(tmp_path, monkeypatch):
    root = _project(tmp_path)
    state = tagging.values(root, read_config(root))
    found = tagging.sources(root, state)
    def broken(*args, **kwargs):
        raise ValueError("Invalid annotation geometry")
    monkeypatch.setattr(tagging, "tag_frame", broken)
    events = []
    result = tagging.run({"state": state, "found": found, "tables": [found["rows"][0]["source"]]}, None,
                         lambda message, progress=None: events.append(message))
    assert result["failed"] == 1 and result["count"] == 0
    assert any("Failed: Invalid annotation geometry" in message for message in events)
