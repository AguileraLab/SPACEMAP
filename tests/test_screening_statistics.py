import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from spacemap_gui.studio import screening as s
from spacemap_gui.studio import screening_statistics as ss
from test_studio_screening import cells, design


def test_scopes_include_all_zones_and_classic_matches_classic():
    from spacemap_gui.tabs.tab8_analysis import feature_screening_core as classic
    c = design(families=['proportion', 'diversity'], widths=[5, 10], correction_scope='classic')
    result, _, _, _, merged = s.compute(cells(), c, {i:.5 for i in cells().image.unique()})
    config = {'screening': {'fdr_mode':'per_family', 'min_patients_per_group':3}}
    endpoint = {'type':'binary', 'label':'Groups', '_group_key':'_group', 'config':config}
    for zone, frame in merged.groupby('zone'):
        frame = frame.rename(columns={'_sample':'patient', 'family':'feature_family', 'feature':'feature_name', 'label':'feature_label'})
        frame = frame.assign(stratum_label='All', zone_width=zone, zone_label=str(zone))
        expected = classic._screen_patient_features_from_merged(frame, config, [endpoint]).set_index('feature_name')
        actual = result[result.zone == zone].set_index('feature')
        np.testing.assert_allclose(actual.loc[expected.index, 'q'], expected.fdr_q, rtol=1e-12)
    for _, frame in result.groupby('family'):
        expected = multipletests(frame.p.fillna(1), method='fdr_bh')[1]
        expected[frame.p.isna()] = np.nan
        np.testing.assert_allclose(frame.q_family, expected)
    assert s.defaults({'alpha':.05})['correction_scope'] == 'global'
    assert s.defaults()['correction_scope'] == 'family'


def test_missing_tests_do_not_shrink_full_search_scopes():
    r = pd.DataFrame({'family':['density']*3+['proportion'], 'zone':[0,1,1,0], 'p':[.01,np.nan,.05,.8]})
    out = ss.adjust(r, {'correction':'fdr_bh','correction_scope':'family'})
    np.testing.assert_allclose(out.loc[0,['q','q_classic','q_global']].to_numpy(float), [.03,.01,.04])
    assert np.isnan(out.loc[1,'q'])


def test_vectorized_mannwhitney_matches_individual_scipy_auto_with_missing_ties():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16,25))
    x[:,::3] = np.round(x[:,::3])
    x[:3,::4] = np.nan
    x[:,1] = 0
    x[:,2] = np.nan
    groups = np.array(['Control']*8+['Case']*8)
    effect, p = ss._fast_mannwhitney(x, groups, design())
    for i in range(x.shape[1]):
        a, b = x[:8,i], x[8:,i]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if not len(a) or not len(b):
            assert np.isnan(p[i])
        else:
            assert p[i] == stats.mannwhitneyu(a,b,alternative='two-sided').pvalue
            assert effect[i] == np.median(b)-np.median(a)


def test_resampling_reproducible_and_does_not_change_primary_tests():
    c = design(families=['proportion'], widths=[5,10], correction_scope='family', resampling='off')
    result, _, _, _, merged = s.compute(cells(), c, {i:.5 for i in cells().image.unique()})
    for mode in ['subsample','bootstrap']:
        cfg = dict(c, resampling=mode, resamples=24)
        out = ss.resample(result, merged, cfg, s.infer)
        again = ss.resample(result, merged, cfg, s.infer)
        pd.testing.assert_frame_equal(out, again)
        pd.testing.assert_frame_equal(out[result.columns], result)
        assert (out.nearby_recovery >= out.recovery).all()
        assert out.valid_resamples.max() <= 24
        if mode == 'subsample':
            assert out.ci_low.isna().all()
        else:
            assert out.ci_low.notna().any()
            assert (out.ci_high.dropna() >= out.ci_low.dropna()).all()


def test_paired_bootstrap_keeps_subject_pairs_together():
    c = design(families=['proportion'], resampling='bootstrap', resamples=20)
    c['endpoint']['paired'] = True
    frame = pd.DataFrame([{'_sample':str(i),'_group':g,'family':'proportion','feature':'["proportion", "A"]',
                          'label':'A','zone':0,'value':i+(2 if g=='Case' else 0),'outcome':g}
                         for i in range(5) for g in ['Control','Case']])
    outcomes = frame[['_sample','_group','outcome']]
    r, merged = s.infer(frame.drop(columns='outcome'), outcomes, c)
    out = ss.resample(r,merged,c,s.infer)
    assert out.iloc[0].ci_low == out.iloc[0].ci_high == 2
    assert out.iloc[0].valid_resamples == 20


def test_findings_beyond_shortlist_are_available_and_linkable(tmp_path):
    frame = cells();table = tmp_path/'cells.parquet';frame.to_parquet(table,index=False)
    c = design(families=['proportion'],widths=list(range(1,31)))
    settings = s.prepare(tmp_path,table,c,frame.image.unique().tolist())
    saved = s.job(settings, None, lambda *args:None)
    assert len(saved['tests']) == 90 and len(saved['hits']) == 30
    row = next(r for r in saved['tests'] if not any(h['feature']==r['feature'] and h['zone']==r['zone'] for h in saved['hits']) and r['n_samples']>0)
    detail = s.finding(tmp_path,saved['run'],row['feature'],row['zone'])
    assert detail['observations'] and detail['feature'] == row['feature']
    source = pd.read_csv(saved['folder']+'/tables/significant_features.csv')
    assert len(source) == saved['significant']
    # Older runs acquire their original full tests on read, without being edited.
    path = __import__('pathlib').Path(saved['folder'])/'result.json'
    del saved['tests'];path.write_text(json.dumps(saved));before = path.read_bytes()
    assert len(s.result(tmp_path,saved['run'])['tests']) == 90
    assert path.read_bytes() == before
