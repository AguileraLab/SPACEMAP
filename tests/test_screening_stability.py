import json
from pathlib import Path

import numpy as np
import pandas as pd

from spacemap_gui.studio import screening_stability as st
from test_studio_screening import design


def example():
    rows = []
    for label, qs, effects, recovery in [
        ('CD4', [.01,.02,.03], [4,5,6], .90),
        ('Treg', [.01,.02,.4], [2,3,2], .99),
        ('Mixed', [.01,.01,.01], [2,-2,2], .60),
        ('Nominal', [.8,.8,np.nan], [-1,-1,np.nan], 0),
    ]:
        for zone in range(3):
            rows.append(dict(family='density', feature=json.dumps(['density',label]), label=label, zone=zone,
                p=.01 if np.isfinite(qs[zone]) else np.nan, q=qs[zone], effect=effects[zone], n_samples=20,
                n_a=10,n_b=10,resampling='subsample',resamples=100,valid_resamples=100,
                recovery=recovery if np.isfinite(qs[zone]) else np.nan, nearby_recovery=min(1,recovery+.05),status='tested'))
    c=design(families=['density'],widths=[50,100,150],resampling='subsample',correction_scope='family')
    c['endpoint'].update(group_a='CR',group_b='PR')
    return pd.DataFrame(rows),c


def test_ranking_uses_consistent_fdr_support_and_exact_recovery():
    data,c=example()
    before=data.copy(deep=True)
    found=st.build(data,c)
    pd.testing.assert_frame_equal(data,before)
    assert [r['label'] for r in found['rows']] == ['CD4','Treg','Mixed','Nominal']
    assert found['rows'][0]['supported_zones'] == 3
    assert found['rows'][0]['median_recovery'] == .9
    assert found['rows'][2]['fdr_zones'] == 3 and found['rows'][2]['supported_zones'] == 2
    assert found['rows'][2]['mixed_fdr_directions']
    assert found['rows'][-1]['cells'][0]['state'] == 'nominal'
    assert found['rows'][-1]['cells'][0]['recovery'] == 0
    assert found['rows'][-1]['cells'][-1]['state'] == 'missing'
    assert found['rows'][-1]['cells'][-1]['recovery'] is None
    assert found['rows'][0]['cells'][0]['direction'] == 'Higher in PR'
    exported=pd.read_csv(__import__('io').BytesIO(st.csv_bytes(found)))
    assert len(exported)==len(data)
    assert exported.loc[exported.label=='CD4','q'].tolist()==[.01,.02,.03]


def test_directions_respect_hazard_ratio_null_and_change_roles():
    _,c=example()
    assert st.direction(dict(c['endpoint'],type='survival'),.7) == (-1,'↓ hazard','Higher feature values associated with lower event hazard')
    assert st.direction(dict(c['endpoint'],type='survival'),1)[0] == 0
    assert st.direction(dict(c['endpoint'],type='survival'),2)[0] == 1
    assert st.direction(dict(c['endpoint'],type='continuous',value_col='Score'),-.3)[2]=='Negative association with Score'
    assert st.direction(dict(c['endpoint'],change_col='Visit'),-2)[2]=='Larger after-minus-before change in CR'
    assert st.direction(dict(c['endpoint'],type='describe'),5)[1]=='—'


def test_saved_comparisons_stay_separate_and_rings_have_exact_bounds():
    data,c=example()
    c['zone_mode']='rings'
    e2=dict(c['endpoint'],group_a='Untreated',group_b='Treated')
    all_rows=pd.concat([data.assign(comparison='response'),data.assign(comparison='treatment')],ignore_index=True)
    plans=[{'id':'response','endpoint':c['endpoint'],'stratum':{'visit':'Before'}},
           {'id':'treatment','endpoint':e2,'stratum':{}}]
    found=st.build(all_rows,c,plans)
    assert len(found['rows']) == 8
    assert [z['label'] for z in found['zones']] == ['0–50 µm','50–100 µm','100–150 µm']
    selected=st.select(found,comparison='treatment',search='CD4',only_fdr=True)
    assert selected['total']==1 and selected['rows'][0]['cells'][0]['short']=='Treated'
    assert 'visit=Before' in next(r['name'] for r in found['rows'] if r['comparison']=='response')


def test_group_colours_and_bootstrap_intervals_survive_overview_export():
    data,c=example()
    data['resampling']='bootstrap'
    data['ci_low'],data['ci_high']=1.5,8.0
    c['resampling']='bootstrap'
    found=st.build(data,c,palette={'PR':'#123456'})
    cell=found['rows'][0]['cells'][0]
    assert cell['group']=='PR' and cell['group_colour']=='#123456'
    assert cell['ci_low']==1.5 and cell['ci_high']==8.0
    exported=pd.read_csv(__import__('io').BytesIO(st.csv_bytes(found)))
    assert exported.loc[exported.label=='CD4','ci_low'].eq(1.5).all()


def test_old_runs_with_no_resampling_keep_missing_recovery_and_export(tmp_path):
    data,c=example()
    data=data.drop(columns=['resampling','recovery','nearby_recovery','resamples','valid_resamples'])
    c['resampling']='off'
    found=st.build(data,c)
    assert all(cell['recovery'] is None for r in found['rows'] for cell in r['cells'])
    assert st.select(found,search='not present')['rows']==[]
    for extension in ('png','svg','pdf'):
        image=st.figure_bytes(st.select(found),extension)
        assert len(image)>1000
        (tmp_path/('overview.'+extension)).write_bytes(image)


def test_filters_reuse_completed_overview_and_changed_results_invalidate(tmp_path, monkeypatch):
    data,c=example()
    saved=dict(settings=c, folder=str(tmp_path), revision='1', tests=data.to_dict('records'), hits=[])
    result_path=tmp_path/'result.json'
    result_path.write_text('{}')
    monkeypatch.setattr(st.screening,'result',lambda *a:saved)
    original=st.build
    calls=[]
    def measured(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(st,'build',measured)
    first=st.load(tmp_path,'run')
    assert st.select(st.load(tmp_path,'run'),search='CD4')['total']==1
    assert len(calls)==1
    saved['tests'][0]['q']=.9
    result_path.write_text('{"edited":true}')
    second=st.load(tmp_path,'run')
    assert len(calls)==2
    assert next(r for r in first['rows'] if r['label']=='CD4')['cells'][0]['q']==.01
    assert next(r for r in second['rows'] if r['label']=='CD4')['cells'][0]['q']==.9
    assert len(st._overviews)<=2
