"""Cell tagging's annotations: a typed Annotations folder is kept; imported GeoJSON can be each table's annotation."""
import json

from spacemap_gui.studio import tagging
from spacemap_gui.studio.settings import read_config, revision


def test_a_typed_annotations_folder_is_kept(tmp_path):
    root = tmp_path / "study"
    (root / "outputs/processed_annotations/stage0_tissue_annotation").mkdir(parents=True)
    (root / "outputs/processed_annotations/new").mkdir(parents=True)
    config = {"project_name": "study", "tab_configurations": {},
              "directory_mappings": {tagging.SHARED_GEOJSON_KEY: "${PROJECT_ROOT}/outputs/processed_annotations/stage0_tissue_annotation"}}
    (root / "study.json").write_text(json.dumps(config))
    before = tagging.values(root, read_config(root))
    assert before[tagging.tab5.GEOJSON_DIR_KEY].endswith("stage0_tissue_annotation")
    after = tagging.save(root, {tagging.tab5.GEOJSON_DIR_KEY: str(root / "outputs/processed_annotations/new")}, revision(before))
    assert after[tagging.tab5.GEOJSON_DIR_KEY].endswith("processed_annotations/new")


def test_imported_geojson_is_a_table_annotation(tmp_path):
    folder = tmp_path / "outputs/studio/imported_annotations/abc"
    folder.mkdir(parents=True)
    copy = folder / "1234abcd_P11_roi_1_merged.geojson"
    copy.write_text("{}")
    exports = {"P11_roi_1": str(tmp_path / "export.geojson"), tagging.IMPORTED: {"P11_roi_1": [str(copy)]}}
    state = {tagging.tab5.GEOJSON_DIR_KEY: str(tmp_path), tagging.USE_EXPORT_KEY: True}
    assert tagging.annotation_path({**state, tagging.SOURCE_KEY: "imported"}, "P11_roi_1", None, exports) == (str(copy), False)
    assert tagging.annotation_path(state, "P11_roi_1", None, exports) == (exports["P11_roi_1"], True)
    assert tagging.annotation_path(state, "P11_roi_1", "imported:" + copy.name, exports) == (str(copy), False)
    assert tagging._shown_name(copy) == "P11_roi_1_merged.geojson"
