"""Repeated observations stay within their independent subject throughout scans."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from spacemap_gui.studio import screening as s, screening_comparisons as sc, screening_statistics as ss
from test_studio_screening import cells, design


def visits():
    frame = cells()
    frame['visit'] = frame.image.str.rsplit('_', n=1).str[-1].map({'0':'Before', '1':'After'})
    frame['treatment'] = frame.subject.str[-1].astype(int).map(lambda i: 'X' if i % 2 == 0 else 'Y')
    # A clear within-subject shift in one outcome group; repeated cell rows are
    # never independent observations for the group comparison.
    frame.loc[(frame.visit == 'After') & (frame.outcome == 'Case'), 'type'] = 'A'
    return frame


def test_plans_strata_pairs_and_changes_preserve_subjects():
    frame = visits()
    c = design(split_by=['visit', 'treatment'], split_depth=2,
        change={'enabled':True, 'column':'visit', 'before':'Before', 'after':'After'},
        extra_comparisons=[{'group_col':'visit', 'all_pairs':True, 'paired':True}])
    plans = sc.plan(frame, c)
    assert len(plans) == 15  # nine response, three paired visit, three delta
    assert sum(p['status'] == 'ready' for p in plans) == 7
    primary = plans[0]
    assert primary['samples'] == {'Control':4, 'Case':4}
    matched = next(p for p in plans if p['endpoint']['paired'] and not p['stratum'])
    assert matched['samples']['matched'] == 8
    delta = next(p for p in plans if p['endpoint'].get('change_col') and not p['stratum'])
    assert delta['samples']['complete_by_group'] == {'Control':4, 'Case':4}
    assert all('visit' not in p['stratum'] for p in plans if p['endpoint'].get('change_col') or p['endpoint']['paired'])
    with pytest.raises(ValueError, match='independent sample ID'):
        sc.plan(frame, design(extra_comparisons=[{'group_col':'subject', 'all_pairs':True}]))


def test_delta_calculates_visit_values_first_and_requires_both():
    frame = visits()
    frame = frame[~((frame.subject == 'S0') & (frame.visit == 'After'))]
    c = design(widths=[100], min_samples=3)
    c['endpoint'].update(change_col='visit', change_before='Before', change_after='After')
    result, samples, regions, audit, merged = s.compute(frame, c, dict.fromkeys(frame.image.unique(), .5))
    # Use the actual feature key rather than making its serialization a contract.
    a = samples[samples.label == 'A']
    assert len(a) == 8 and a.loc[a._sample == 'S0','value'].isna().all()
    assert np.allclose(a.loc[a._group == 'Case', 'value'], 100-100/3)
    assert np.allclose(a.loc[(a._group == 'Control') & (a._sample != 'S0'), 'value'], 0)
    row = result[result.label == 'A'].iloc[0]
    assert row.n_samples == 7 and row.n_a == 3 and row.n_b == 4
    assert row.p == stats.mannwhitneyu([0]*3, [100-100/3]*4, alternative='two-sided').pvalue
    payload = s._display_result({'values':c,'table':'fixture','source':{},'mpp':{}}, Path('/tmp/unused'), result, merged, regions, audit, [])
    hit = next(r for r in payload['hits'] if r['label'] == 'A')
    assert hit['highest'] == hit['lowest'] == []
    assert all([v['name'] for v in p['visits']] == ['Before','After'] for p in hit['observations'])
    assert all(len(p['images']) == 2 for p in hit['observations'])


def test_shared_measurements_are_reused_without_outcome_dependent_null(tmp_path, monkeypatch):
    frame = visits()
    c = design(widths=[100])
    seen = []
    original = s._measure
    def measured(part, *args):
        seen.append(part._group.unique().tolist())
        return original(part, *args)
    monkeypatch.setattr(s, '_measure', measured)
    cache = sc.MeasurementCache(tmp_path)
    s.compute(frame, c, dict.fromkeys(frame.image.unique(), .5), measurement_cache=cache)
    other = deepcopy(c)
    other['endpoint'].update(group_col='visit', group_a='Before', group_b='After', paired=True)
    s.compute(frame, other, dict.fromkeys(frame.image.unique(), .5), measurement_cache=cache)
    assert cache.hits == cache.misses == 16
    assert seen == [['Measurement']] * 16


def test_saved_multiple_comparisons_joint_fdr_and_links(tmp_path, monkeypatch):
    from spacemap_gui.studio import screening_figures
    monkeypatch.setattr(screening_figures, 'save', lambda *args: [])
    frame = visits()
    table = tmp_path/'cells.parquet'
    frame.to_parquet(table, index=False)
    c = design(widths=[100], split_by=['visit','treatment'], split_depth=2, correction_scope='family',
        change={'enabled':True, 'column':'visit', 'before':'Before', 'after':'After'},
        extra_comparisons=[{'group_col':'visit', 'all_pairs':True, 'paired':True}])
    settings = s.prepare(tmp_path, table, c, frame.image.unique().tolist())
    result = s.job(settings, None, lambda *a: None)
    out = Path(result['scan_folder'])
    full = pd.read_csv(out/'tables/all_comparisons.csv')
    expected = multipletests(full.p.fillna(1), method='fdr_bh')[1]
    expected[full.p.isna()] = np.nan
    np.testing.assert_allclose(full.q, expected)
    assert len(full) == 45 and full.comparison.nunique() == 15
    assert full.loc[full.p.isna(), 'q'].isna().all()
    assert json.loads((out/'design.json').read_text())['measurement_cache'] == {'computed':16, 'reused':64}
    assert all(p['status'] in ('complete','skipped') for p in result['comparisons'])
    delta = next(p for p in result['comparisons'] if p['endpoint'].get('change_col') and p['status'] == 'complete')
    selected = s.result(tmp_path, result['run'], delta['id'])
    assert selected['settings']['endpoint']['change_col'] == 'visit'
    assert selected['scan_settings'] == c
    delta_hit = selected['hits'][0]
    assert s.context_cells(tmp_path, table, result['run'], 'S0_0', delta_hit['feature'], 0,
                           comparison=delta['id'], sample='S0', group='Control', occasion='After')['labels'] == []
    timepoint = next(p for p in result['comparisons'] if p['stratum'] == {'visit':'Before'})
    selected = s.result(tmp_path, result['run'], timepoint['id'])
    hit = selected['hits'][0]
    linked = s.context_cells(tmp_path, table, result['run'], 'S0_0', hit['feature'], hit['zone'], comparison=timepoint['id'])
    assert len(linked['labels']) > 0
    assert s.context_cells(tmp_path, table, result['run'], 'S0_1', hit['feature'], hit['zone'], comparison=timepoint['id'])['labels'] == []


def test_joint_resampling_draws_same_subjects_in_overlapping_questions(monkeypatch):
    frame = visits()
    c = design(widths=[100], resampling='bootstrap', resamples=20, min_samples=2)
    r, _, _, _, merged = s.compute(frame, dict(c, resampling='off'), dict.fromkeys(frame.image.unique(), .5))
    all_results = ss.adjust(pd.concat([r.assign(comparison='one'), r.assign(comparison='two')], ignore_index=True), c)
    result = ss.resample_joint(all_results, {'one':(merged,c), 'two':(merged,c)}, c, s.infer)
    cols = ['recovery','nearby_recovery','ci_low','ci_high','valid_resamples']
    np.testing.assert_allclose(result[result.comparison == 'one'][cols], result[result.comparison == 'two'][cols], equal_nan=True)
    # Classic scope separates the declared question; family scope includes both.
    assert (result.n_tests_family == 6).all() and (result.n_tests_classic == 3).all()


def test_area_csv_reuses_unambiguous_physical_areas_for_other_questions(tmp_path):
    frame = visits()
    areas = frame[['subject','outcome','image']].drop_duplicates().rename(columns={'subject':'sample','outcome':'group'})
    areas = areas.assign(region='All', zone=0, area_mm2=.01)
    area_path = tmp_path/'areas.csv'
    areas.to_csv(area_path,index=False)
    c = design(widths=[100], families=['density'], area_table=str(area_path))
    c['options']['density']['area_source'] = 'csv'
    _, _, original, _, _ = s.compute(frame, c, dict.fromkeys(frame.image.unique(), .5))
    c['endpoint'].update(group_col='visit', group_a='Before', group_b='After', paired=True)
    _, _, changed, _, _ = s.compute(frame, c, dict.fromkeys(frame.image.unique(), .5))
    cols=['_image','feature','zone']
    np.testing.assert_allclose(original.set_index(cols).sort_index().value, changed.set_index(cols).sort_index().value)
