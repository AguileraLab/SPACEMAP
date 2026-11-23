"""Incremental subtyping must preserve other parents through clustering, naming and commit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels, pending_columns
from spacemap_gui.studio import commands, jobs, leiden, subphenotypes, tasks


@pytest.fixture(params=['csv', 'parquet'])
def study(tmp_path, monkeypatch, request):
    monkeypatch.setenv('SPACEMAP_FAST_COPY', 'off')
    import spacemap.ph_subpheno_utils as clustering
    # Test publication of a known partition, independently of stochastic clustering.
    monkeypatch.setattr(clustering, 'subcluster', lambda matrix, *a, **kw: np.arange(len(matrix)) % 2)
    rng = np.random.default_rng(3)
    parts = []
    for image in ['R1', 'R2']:
        for parent, count in [('Epithelial', 120), ('Fibroblast', 160), ('Macrophage', 40)]:
            parts.append(pd.DataFrame({'image': image, 'parent': parent,
                'region': (np.arange(count) < 120).astype(int),
                'M1': rng.uniform(1, 100, count), 'M2': rng.uniform(1, 100, count)}))
    frame = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=4).reset_index(drop=True)
    frame.insert(0, 'cell_key', [f'cell-{i}' for i in range(len(frame))])
    table = tmp_path / ('cells.' + request.param)
    if request.param == 'csv':
        frame.to_csv(table, index=False)
    else:
        frame.to_parquet(table, index=False)
    monkeypatch.setattr(commands.registry, 'session_project', lambda _: {'root': str(tmp_path)})
    return tmp_path, table


def split(root, table, parents, monkeypatch, *, images=None, filters=(), legacy=False):
    config = {**subphenotypes.DEFAULTS, 'parent_col': 'parent', 'parent_values': parents,
              'markers': ['M1', 'M2'], 'filters': list(filters), 'use_gpu': False}
    settings = {'table': str(table), 'project_root': str(root), 'markers': ['M1', 'M2'],
                'output_col': 'phenotype_sub_clusters', 'phenotype_col': 'phenotype_sub',
                'cell_id_col': 'cell_key', 'roi_column': 'image', 'selected_images': images,
                'config': config, 'mode': 'run',
                'overwrite_columns': ['phenotype_sub_clusters', 'phenotype_sub']}
    settings['revision'] = tasks.source_revision(settings)
    folder = root / ('run-' + '-'.join(parents))
    result = subphenotypes.run(settings, folder, lambda *a: None)
    if legacy:
        result.pop('scoped_membership', None)
    record = {'id': folder.name, 'kind': 'subphenotype', 'status': 'done', 'result': result}
    (folder / 'request.json').write_text(json.dumps({'settings': settings}))
    monkeypatch.setattr(jobs, 'status', lambda *a: record)
    monkeypatch.setattr(jobs, 'directory', lambda *a: folder)
    monkeypatch.setattr(jobs, 'recent', lambda *a, **kw: [record])
    monkeypatch.setattr(jobs, 'history', lambda *a, **kw: iter([record]))
    return result, settings, folder


def name(root, table, result, monkeypatch):
    names = {item['name']: 'Named ' + item['name'] for item in result['clusters']}
    captured = []
    def start(root, kind, settings, **kwargs):
        assert kind == 'name_clusters'
        captured.append(settings)
        return leiden.name_clusters(settings, Path(root) / 'names', lambda *a: None)
    monkeypatch.setattr(jobs, 'start', start)
    commands._leiden('test', 'name_clusters', {'job_id': 'test', 'names': names}, {}, str(table), 'request')
    assert captured and captured[0]['row_scoped']
    return names, captured[0]


@pytest.mark.parametrize('committed', [False, True])
@pytest.mark.parametrize('image_selection', [None, ['R1']])
def test_overwrite_and_naming_preserve_other_cells(study, monkeypatch, committed, image_selection):
    root, table = study
    first, _, _ = split(root, table, ['Epithelial'], monkeypatch)
    name(root, table, first, monkeypatch)
    original = dataio.load(table)
    assert original.loc[original.parent.eq('Epithelial'), 'phenotype_sub'].str.startswith('Named Epithelial_').all()
    # Existing results outside the next run include the same cluster IDs and manual edits.
    for col, value in [('phenotype_sub', 'Earlier fibroblast'), ('phenotype_sub_clusters', 'Fibroblast_0')]:
        original.loc[original.parent.eq('Fibroblast'), col] = value
    for col in ['phenotype_sub', 'phenotype_sub_clusters']:
        original.loc[original.parent.eq('Macrophage'), col] = 'Keep rare population'
    pending_columns.save(table, {c: original[c] for c in ['phenotype_sub', 'phenotype_sub_clusters']},
                         anchor_column='cell_key', anchor_values=original.cell_key)
    epi = original.index[original.parent.eq('Epithelial')][0]
    labels.append(table, [{'row': int(epi), 'column': 'phenotype_sub', 'value': 'Manual epithelial label'}])
    if committed:
        pending_columns.commit(table, rebuild_fast_copy=False)
    before = dataio.load(table)
    raw_before = table.read_bytes()
    result, settings, folder = split(root, table, ['Fibroblast', 'Macrophage'], monkeypatch,
        images=image_selection, filters=[{'column': 'region', 'op': 'in', 'values': ['1']}])
    affected = before.parent.eq('Fibroblast') & before.region.eq(1)
    if image_selection:
        affected &= before.image.isin(image_selection)
    assert result['cells'] == int(affected.sum())
    after_split = dataio.load(table)
    pd.testing.assert_frame_equal(after_split.loc[~affected], before.loc[~affected])
    assert after_split.loc[affected, 'phenotype_sub'].str.startswith('Fibroblast_').all()
    assert after_split.loc[affected, 'phenotype_sub_clusters'].str.startswith('Fibroblast_').all()
    names, naming = name(root, table, result, monkeypatch)
    after_names = dataio.load(table)
    pd.testing.assert_frame_equal(after_names.loc[~affected], before.loc[~affected])
    assert after_names.loc[affected, 'phenotype_sub'].equals(after_names.loc[affected, 'phenotype_sub_clusters'].map(names))
    pd.testing.assert_frame_equal(after_names.drop(columns=['phenotype_sub', 'phenotype_sub_clusters']),
                                  before.drop(columns=['phenotype_sub', 'phenotype_sub_clusters']))
    assert table.read_bytes() == raw_before  # Still staged until explicit Write into table.
    # Clearing names resets only this run, never names assigned by another run.
    naming.update(names={}, revision=tasks.source_revision(naming))
    leiden.name_clusters(naming, folder, lambda *a: None)
    reset = dataio.load(table)
    pd.testing.assert_frame_equal(reset.loc[~affected], before.loc[~affected])
    assert reset.loc[affected, 'phenotype_sub'].equals(reset.loc[affected, 'phenotype_sub_clusters'])
    with pytest.raises(ValueError, match='Overwrite existing results'):
        tasks.validate_table({**settings, 'overwrite_columns': []})
    pending_columns.commit(table, rebuild_fast_copy=False)
    on_disk = pd.read_parquet(table) if table.suffix == '.parquet' else pd.read_csv(table)
    pd.testing.assert_frame_equal(on_disk, reset, check_dtype=False)


def test_legacy_naming_and_missing_membership_are_scoped(study, monkeypatch):
    root, table = study
    result, _, folder = split(root, table, ['Epithelial'], monkeypatch, legacy=True)
    _, naming = name(root, table, result, monkeypatch)
    before = dataio.load(table)
    assert before.loc[before.parent.ne('Epithelial'), 'phenotype_sub'].isna().all()
    naming['subphenotype_run']['membership_file'] = str(folder / 'missing.parquet')
    naming['revision'] = tasks.source_revision(naming)
    with pytest.raises(ValueError, match='membership is missing'):
        leiden.name_clusters(naming, folder, lambda *a: None)
    pd.testing.assert_frame_equal(dataio.load(table), before)
