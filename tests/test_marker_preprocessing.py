"""Preprocessing is fitted to the selected population, with explicit image scope."""
import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap.marker_preprocessing import transform
from spacemap.ph_subpheno_utils import build_feature_matrix
from spacemap_gui.studio import subphenotypes, tasks


def test_raw_transformed_once_corrected_preserved_and_source_unchanged():
    raw = np.array([-5., 0., 5., 100.])
    frame = pd.DataFrame({'Ki67': raw, 'Ki67_corr': np.arcsinh(raw / 5)})
    original = frame.copy()
    result = transform(frame, list(frame))
    np.testing.assert_allclose(result.Ki67, result.Ki67_corr)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(transform(frame, list(frame), 'stored'), frame)
    missing = transform(pd.DataFrame({'M': [np.inf, 'bad', None]}), ['M'])
    assert missing.isna().all().all()


def test_slide_normalization_pools_rois_and_reports_fit():
    frame = pd.DataFrame({'slide': ['A'] * 4 + ['B'] * 4,
                          'roi': ['a1', 'a1', 'a2', 'a2', 'b1', 'b1', 'b2', 'b2'],
                          'M': [1, 2, 9, 10, 101, 102, 109, 110]})
    values, fitted = build_feature_matrix(frame, ['M'], batch_col='slide', return_info=True)
    np.testing.assert_allclose(values[:4], values[4:])
    assert np.mean(values[:2]) < 0 and np.mean(values[2:4]) > 0
    assert [f['center'] for f in fitted] == [5.5, 105.5]
    global_values = build_feature_matrix(frame, ['M'])
    assert np.mean(global_values[:4]) < 0 and np.mean(global_values[4:]) > 0
    json.dumps(fitted, allow_nan=False)


def test_robust_scaling_keeps_rare_signal_and_imputes_missing_center():
    frame = pd.DataFrame({'M': [0.] * 8 + [10., np.nan]})
    values, fitted = build_feature_matrix(frame, ['M'], normalization='robust', return_info=True)
    assert values[8, 0] > 0 and values[9, 0] == 0 and fitted[0]['iqr_fallback']
    unscaled = build_feature_matrix(frame, ['M'], normalization='none')
    assert unscaled[8, 0] == 10 and unscaled[9, 0] == 0


def test_older_figure_transformation_becomes_shared_setting(tmp_path):
    path = subphenotypes.cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'figure_options': {'transform': 'stored', 'cofactor': 20}}))
    config = subphenotypes.values(tmp_path)
    assert config['transform'] == 'stored' and config['cofactor'] == 20
    from spacemap_gui.studio import marker_figures
    table = tmp_path/'cells.parquet'
    pd.DataFrame({'phenotype': ['Epithelial'], 'M': [100]}).to_parquet(table)
    config.update(parent_col='phenotype', parent_values=['Epithelial'], markers=['M'],
                  transform='auto', cofactor=10)
    settings = marker_figures.prepare(tmp_path, table, config)
    assert settings['options']['transform'] == 'auto' and settings['options']['cofactor'] == 10


@pytest.mark.parametrize('groups', [['A', None], ['A', '']])
def test_missing_normalization_group_is_actionable(groups):
    with pytest.raises(ValueError, match='missing normalization groups'):
        build_feature_matrix(pd.DataFrame({'slide': groups, 'M': [1, 2]}), ['M'], batch_col='slide')


def test_selected_images_limit_fitting_and_preserve_other_results(tmp_path, monkeypatch):
    import spacemap.ph_subpheno_utils as clustering
    root = tmp_path / 'study'; root.mkdir()
    path = root / 'cells.parquet'
    n = 120
    frame = pd.DataFrame({'cell_id': np.arange(2*n), 'image': ['roi1']*n+['roi2']*n,
        'slide': ['one']*n+['two']*n, 'phenotype_parent': ['Epithelia']*(2*n),
        'M': np.r_[np.full(n, 1e8), np.arange(n)], 'N_corr': np.r_[np.full(n, 1e8), np.arange(n)/10],
        'phenotype_sub_clusters': ['prior cluster']*(2*n), 'phenotype_sub': ['prior name']*(2*n)})
    frame.to_parquet(path, index=False)
    labels.append(path, [{'row': 0, 'column': 'phenotype_sub', 'value': 'Manual edit'}])
    config = {**subphenotypes.DEFAULTS, 'parent_col': 'phenotype_parent', 'parent_values': ['Epithelia'],
              'markers': ['M', 'N_corr'], 'chosen_markers': {'Epithelia': ['M', 'N_corr']},
              'normalization_col': 'slide', 'use_gpu': False}
    settings = dict(table=str(path), project_root=str(root), config=config, mode='run',
                    output_col='phenotype_sub_clusters', phenotype_col='phenotype_sub',
                    cell_id_col='cell_id', markers=config['markers'], roi_column='image',
                    selected_images=['roi2'], overwrite_columns=['phenotype_sub', 'phenotype_sub_clusters'])
    settings['revision'] = tasks.source_revision(settings)
    received = []
    def cluster(matrix, *args, **kwargs):
        received.append(matrix.copy())
        assert matrix.shape == (n, 2)
        np.testing.assert_allclose(matrix.mean(axis=0), 0, atol=1e-6)
        return np.arange(n) % 2
    monkeypatch.setattr(clustering, 'subcluster', cluster)
    selection = subphenotypes.selection(path, config, settings)
    assert selection['cells'] == n
    discovery = subphenotypes.run({**settings, 'mode': 'discover'}, tmp_path/'discover', lambda *a: None)
    assert discovery['discovery']['Epithelia']['cells'] == n
    result = subphenotypes.run(settings, tmp_path/'run', lambda *a: None)
    saved = dataio.load(path)
    assert result['cells'] == n and result['selection']['images'] == ['roi2']
    assert saved.loc[:n-1, 'phenotype_sub_clusters'].eq('prior cluster').all()
    assert saved.loc[0, 'phenotype_sub'] == 'Manual edit'
    assert saved.loc[1:n-1, 'phenotype_sub'].eq('prior name').all()
    assert saved.loc[n:, 'phenotype_sub_clusters'].str.startswith('Epithelia_').all()
    fit = json.loads((tmp_path/'run/preprocessing.json').read_text())['fitted']['Epithelia']
    assert all(f['group'] == 'two' and f['finite'] == n for f in fit)
    assert fit[0]['center'] == pytest.approx(np.arcsinh(np.arange(n)/5).mean())
    assert fit[1]['center'] == pytest.approx(np.arange(n).mean()/10)
    np.testing.assert_array_equal(pd.read_parquet(path)[['M', 'N_corr']], frame[['M', 'N_corr']])
