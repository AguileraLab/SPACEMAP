"""After re-segmenting: Cell tagging reads the new tables, and older tagged or combined tables are flagged."""

import json
import os
import time

from spacemap_gui.studio import freshness, tagging
from spacemap_gui.studio.settings import read_config
from spacemap_gui.tabs import cell_tagging_ui as tab5


def _project(tmp_path):
    root = tmp_path / "study"
    new = root / "outputs/segmentation_new"
    new.mkdir(parents=True)
    old = root / "outputs/segmentation"
    old.mkdir(parents=True)
    for folder in (old, new):
        for roi in ("A_roi_1", "B_roi_1"):
            (folder / f"{roi}.ome_cell_features.csv").write_text("label,x_centroid,y_centroid\n1,1,1\n")
    (root / "study.json").write_text(json.dumps({"directory_mappings": {"tab4_output_dir": "${PROJECT_ROOT}/outputs/segmentation_new"},
                                                 "tab_configurations": {}}))
    return root, new


def test_cell_tagging_follows_the_segmentation_output_unless_a_folder_is_picked(tmp_path):
    root, new = _project(tmp_path)
    state = tagging.values(root, read_config(root))
    assert state[tagging.FOLLOW_KEY] and state[tab5.CSV_DIR_KEY] == str(new)
    picked = root / "outputs/segmentation"
    tagging.save(root, {tab5.CSV_DIR_KEY: "outputs/segmentation"}, __import__("spacemap_gui.studio.settings", fromlist=["x"]).revision(state))
    state = tagging.values(root, read_config(root))
    assert not state[tagging.FOLLOW_KEY] and state[tab5.CSV_DIR_KEY] == str(picked)


def test_legacy_table_age_does_not_claim_it_uses_current_cells(tmp_path):
    root, new = _project(tmp_path)
    config = read_config(root)
    state = tagging.values(root, config)
    tagged = root / state[tab5.OUTPUT_DIR_KEY] if not os.path.isabs(state[tab5.OUTPUT_DIR_KEY]) else __import__("pathlib").Path(state[tab5.OUTPUT_DIR_KEY])
    tagged.mkdir(parents=True, exist_ok=True)
    earlier = time.time() - 3600
    for roi in ("A_roi_1", "B_roi_1"):
        sample = tab5.sample_id_from_csv(f"{roi}.ome_cell_features.csv", "")
        path = tagged / f"{sample}{state[tab5.OUTPUT_SUFFIX_KEY]}.csv"
        path.write_text("label\n1\n")
        if roi == "A_roi_1":
            os.utime(path, (earlier, earlier))  # tagged before the new segmentation
    combined = root / "outputs/batch_corrected/combined.parquet"
    combined.parent.mkdir(parents=True)
    combined.write_text("x")
    os.utime(combined, (earlier, earlier))
    found = freshness.check(root, config, config["directory_mappings"], str(combined))
    assert [s.startswith("A_roi_1") for s in found["tagging"]] == [True]
    assert found["combined"] == [] and found['unverified'] and found["table"] == "combined.parquet"


def test_only_recorded_dependencies_mark_a_combined_result_stale(tmp_path):
    from spacemap_gui.studio.correction import _stamp
    root, new = _project(tmp_path)
    source = next(new.glob('*.csv'))
    combined = root / 'combined.parquet'
    combined.touch()
    combined.with_suffix('.json').write_text(json.dumps({'rois': ['A_roi_1'], 'inputs': {'A_roi_1': _stamp(source)}}))
    config = read_config(root)
    assert freshness.check(root, config, config['directory_mappings'], combined)['combined'] == []
    source.write_text(source.read_text() + '\n')
    changed = freshness.check(root, config, config['directory_mappings'], combined)
    assert changed['combined'] == ['A_roi_1'] and not changed['unverified']
