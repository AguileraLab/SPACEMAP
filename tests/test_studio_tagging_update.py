"""Selective retagging preserves unified tables and both saved formats."""
import csv
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacemap_gui.studio import tagging, tagging_update
from spacemap_gui.studio.settings import read_config
from spacemap_gui.tabs import cell_tagging_ui as tab5
from test_studio_tagging import _project


def test_csv_integer_id_comparison_does_not_round_large_ids():
    assert tagging_update._equal_text('9007199254740992', 9007199254740992)
    assert not tagging_update._equal_text('9007199254740993', 9007199254740992)
    assert tagging_update._equal_text('123.0', 123)


def fixture(root, primary='.parquet', pair=True):
    _project(root)
    tables = root / 'outputs/segmentation/cellpose/features'
    frames = []
    for roi in ['P1_roi_1', 'P2_roi_1']:
        frame = pd.read_csv(tables / f'{roi}_cell_features.csv')
        frame['batch_id'] = roi
        frame['cell_id'] = [f'{roi}:{label}' for label in frame.label]
        frame['Tumor'] = 9
        frame['Tumor_distance'] = 777.0
        frame['M_corr'] = frame.label * .125
        frame['phenotype_ref1'] = '003'
        frame['TLS_distance'] = 8.5
        frames.append(frame)
    original = pd.concat(frames, ignore_index=True)
    # Interleave ROIs to catch accidental concat/reordering.
    original = original.iloc[[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]].reset_index(drop=True)
    combined = root / 'combined'
    combined.mkdir()
    for ext in (['.csv', '.parquet'] if pair else [primary]):
        path = combined / ('cells' + ext)
        if ext == '.csv':
            original.to_csv(path, index=False)
        else:
            table = pa.Table.from_pandas(original, preserve_index=False)
            pq.write_table(table.replace_schema_metadata({**table.schema.metadata,
                b'test_metadata': b'keep me', b'spacemap_column_units': b'{"TLS_distance":"px"}'}), path)
    state = tagging.values(root, read_config(root))
    state.update({tab5.MODE_KEY: tab5.MODE_MERGED, tab5.MERGED_DIR_KEY: str(combined),
                  tab5.MERGED_FILE_KEY: 'cells' + primary, tagging.UPDATE_KEY: True,
                  tab5.CLEAN_PATH_KEY: False, tagging.UNITS_KEY: 'px'})
    found = tagging.sources(root, state)
    rules = tagging.rules(found, state)
    for rule in rules:
        rule.update({'Binary (0/1)': rule['Annotation Label'] == 'Tumor', 'ROI ID': False,
                     'ROI Size': False, 'Distance': False})
    state[tab5.RULES_KEY] = rules
    return {'state': state, 'found': found, 'tables': ['P1_roi_1'], 'root': str(root)}, original


@pytest.mark.parametrize('primary', ['.csv', '.parquet'])
def test_update_both_formats_without_touching_other_fields_or_images(tmp_path, primary):
    settings, old = fixture(tmp_path, primary)
    folder = tmp_path / 'combined'
    original_csv = list(csv.reader((folder / 'cells.csv').open()))
    result = tagging.run(settings, tmp_path / 'job', lambda *a: None)
    expected = old.copy()
    target = expected.batch_id == 'P1_roi_1'
    expected.loc[target, 'Tumor'] = (expected.loc[target, 'x_centroid'] <= 50).astype(int)
    pd.testing.assert_frame_equal(pd.read_parquet(folder / 'cells.parquet'), expected)
    current_csv = list(csv.reader((folder / 'cells.csv').open()))
    column = original_csv[0].index('Tumor')
    for i, (before, after) in enumerate(zip(original_csv, current_csv)):
        assert all(a == b for j, (a, b) in enumerate(zip(before, after)) if j != column or i == 0)
    assert pq.read_schema(folder / 'cells.parquet').metadata[b'test_metadata'] == b'keep me'
    assert result['updated'] and result['cells'] == 6 and result['rows'] == 12 and result['count'] == 2
    backup = Path(result['backup'])
    pd.testing.assert_frame_equal(pd.read_parquet(backup / 'cells.parquet'), old)
    assert list(csv.reader((backup / 'cells.csv').open())) == original_csv
    assert tagging.outputs(settings['state'], settings['found']) == {'P1_roi_1': 'cells' + primary}


@pytest.mark.parametrize('primary', ['.csv', '.parquet'])
def test_new_region_columns_are_missing_for_unprocessed_images(tmp_path, primary):
    settings, old = fixture(tmp_path, primary, pair=False)
    settings['state'][tagging.NAMES_KEY] = {'Tumor|inside': 'new_region'}
    result = tagging.run(settings, tmp_path / 'job', lambda *a: None)
    path = Path(result['outputs'][0])
    current = pd.read_parquet(path) if primary == '.parquet' else pd.read_csv(path, dtype={'phenotype_ref1': str})
    assert current.loc[current.batch_id == 'P2_roi_1', 'new_region'].isna().all()
    assert current.loc[current.batch_id == 'P1_roi_1', 'new_region'].tolist() == [1, 1, 1, 0, 0, 0]
    pd.testing.assert_frame_equal(current[old.columns], old)


@pytest.mark.parametrize('problem', ['reordered_csv', 'pending', 'identity_column', 'empty_selection', 'changed_source', 'changed_annotation'])
def test_invalid_or_concurrent_updates_leave_both_originals_intact(tmp_path, problem):
    settings, _ = fixture(tmp_path)
    folder = tmp_path / 'combined'
    if problem == 'reordered_csv':
        path = folder / 'cells.csv'
        frame = pd.read_csv(path)
        frame.iloc[::-1].to_csv(path, index=False)
    elif problem == 'pending':
        (folder / 'cells.parquet.labels.parquet').touch()
    elif problem == 'identity_column':
        settings['state'][tagging.NAMES_KEY] = {'Tumor|inside': 'cell_id'}
    elif problem == 'empty_selection':
        settings['tables'] = []
    originals = {p: p.read_bytes() for p in [folder / 'cells.csv', folder / 'cells.parquet']}
    changed = False
    def progress(message, fraction=None):
        nonlocal changed
        if problem in ['changed_source', 'changed_annotation'] and not changed and message.startswith('Preparing updated'):
            changed = True
            path = (folder / 'cells.csv') if problem == 'changed_source' else Path(settings['state'][tab5.GEOJSON_DIR_KEY]) / 'P1_roi_1_zones.geojson'
            import os
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000))
    with pytest.raises(ValueError):
        tagging.run(settings, tmp_path / 'job', progress)
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_failure_publishing_second_format_rolls_back_and_retains_backups(tmp_path, monkeypatch):
    import os
    settings, old = fixture(tmp_path)
    replace = os.replace
    def fail_second(source, target):
        if Path(source).name == '1.csv':
            raise OSError('simulated disk failure')
        return replace(source, target)
    monkeypatch.setattr(os, 'replace', fail_second)
    with pytest.raises(OSError, match='disk failure'):
        tagging.run(settings, tmp_path / 'job', lambda *a: None)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'combined/cells.parquet'), old)
    assert len(list((tmp_path / 'combined/.cell_tagging_backups').glob('*/cells.parquet'))) == 1


def test_partial_update_cannot_mix_known_distance_units(tmp_path):
    settings, old = fixture(tmp_path)
    for rule in settings['state'][tab5.RULES_KEY]:
        rule['Distance'] = rule['Annotation Label'] == 'Tumor'
    settings['state'][tagging.NAMES_KEY] = {'Tumor|distance': 'TLS_distance'}
    settings['state'][tagging.UNITS_KEY] = 'um'
    project = tmp_path / (tmp_path.name + '.json')
    config = json.loads(project.read_text()); config['mpp_codex'] = .5
    project.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='keep those units'):
        tagging.run(settings, tmp_path / 'job', lambda *a: None)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'combined/cells.parquet'), old)


def test_changed_parent_membership_requires_explicit_parent_update(tmp_path):
    settings, old = fixture(tmp_path)
    old['tissue'] = 0  # the saved parent does not contain these cells
    old.to_csv(tmp_path / 'combined/cells.csv', index=False)
    old.to_parquet(tmp_path / 'combined/cells.parquet', index=False)
    with pytest.raises(ValueError, match='parent polygons differ'):
        tagging.run(settings, tmp_path / 'job', lambda *a: None)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'combined/cells.parquet'), old)
    for rule in settings['state'][tab5.RULES_KEY]:
        if rule['Annotation Label'] == 'tissue':
            rule['Binary (0/1)'] = True
    tagging.run(settings, tmp_path / 'job2', lambda *a: None)
    current = pd.read_parquet(tmp_path / 'combined/cells.parquet')
    assert current.loc[current.batch_id == 'P1_roi_1', 'tissue'].eq(1).all()
    assert current.loc[current.batch_id == 'P2_roi_1', 'tissue'].eq(0).all()


def test_csv_only_update_remembers_units_across_later_runs(tmp_path):
    settings, _ = fixture(tmp_path, '.csv', pair=False)
    for rule in settings['state'][tab5.RULES_KEY]:
        rule['Distance'] = rule['Annotation Label'] == 'Tumor'
    tagging.run(settings, tmp_path / 'job', lambda *a: None)
    settings['state'][tagging.UNITS_KEY] = 'um'
    project = tmp_path / (tmp_path.name + '.json')
    config = json.loads(project.read_text()); config['mpp_codex'] = .5
    project.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='keep those units'):
        tagging.run(settings, tmp_path / 'job2', lambda *a: None)
