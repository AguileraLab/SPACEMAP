import copy
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from spacemap import flow_statistics as statistics, flow_gating as flow


def study():
    frame=pd.DataFrame([{'patient':str(p),'time':t,'M':p*(1+.1*t)+j*.1,'K':p*p+t+j*.2,'eligible':p!=5}
                        for t in (0,1,2) for p in range(1,7) for j in range(p+2)])
    gates=[{'id':'tissue','kind':'filter','rule':{'column':'eligible','values':['True']}},
           *[{'id':f't{t}','kind':'filter','parent':'tissue','rule':{'column':'time','op':'==','value':t}} for t in (0,1,2)]]
    masks=flow.evaluate(frame,gates)
    design={'groups':[{'id':f'g{t}','population':f't{t}','label':f'Time {t}','colour':'#114499'} for t in (0,1)],
            'features':[{'id':'m','column':'M','metric':'mean','label':'M mean'}, {'id':'k','column':'K','metric':'median','label':'K median'}],
            'group_by':'patient','min_cells':3,'test':'welch','correction':'fdr_bh'}
    return frame,masks,design


def test_patient_aggregation_inheritance_and_global_fdr():
    frame,masks,design=study();result=statistics.run(frame,masks,design)
    assert all(s['N']==5 for s in result['summaries'])
    assert all(s['n_cells']==26 for s in result['summaries'])
    assert all(s['unit']!='5' for s in result['samples'])
    expected=[]
    for f in design['features']:
        arrays=[frame.loc[masks[g['population']][0]].groupby('patient')[f['column']].agg(f['metric']).to_numpy() for g in design['groups']]
        expected.append(stats.ttest_ind(*arrays,equal_var=False).pvalue)
    np.testing.assert_allclose([t['p'] for t in result['tests']],expected)
    np.testing.assert_allclose([t['q'] for t in result['tests']],multipletests(expected,method='fdr_bh')[1])
    # Cell numbers do not masquerade as biological replicates.
    assert len(result['samples'])==20


def test_paired_ids_minimum_missing_and_matched_delta():
    frame,masks,d=study();frame.loc[(frame.time==1)&(frame.patient=='6'),'patient']=None
    d['test']='paired_t';d['heatmap']={'mode':'delta','matched':True,'pairs':[{'id':'change','before':'g0','after':'g1','label':'Treatment'}]}
    result=statistics.run(frame,masks,d)
    assert all(t['matched_N']==4 for t in result['tests'])
    assert any('no patient ID' in x for x in result['notices'])
    m=next(c for c in result['heatmap'] if c['feature']=='m')
    assert m['value']==pytest.approx(.25)
    d['min_cells']=7
    result=statistics.run(frame,masks,d)
    assert all(t['p'] is None for t in result['tests'])


def test_strict_threshold_includes_missing_in_denominator_not_bin_range():
    frame=pd.DataFrame({'M':[0.,2.,3.,np.nan,100.],'patient':['A']*5})
    masks={'root':(np.ones(5,bool),np.ones(5,bool))}
    t={'id':'high','name':'High','x':{'column':'M'},'bounds':[2.]}
    spec={'x':{'column':'M'},'bins':4,'x_range':[0,4],'threshold':t}
    hist=flow.distributions(frame,masks,spec)['series'][0]
    assert hist['n']==5 and hist['above_threshold']==2 and hist['above_percent']==40
    assert hist['outside']==1
    d={'groups':[{'id':'a','population':'root'},{'id':'b','population':'root'}],
       'features':[{'id':'pos','metric':'positive','threshold':t}], 'group_by':'patient','min_cells':1}
    r=statistics.run(frame,masks,d)
    assert r['samples'][0]['value']==40 and r['summaries'][0]['n_valid']==4


def test_proportions_validate_denominator_and_pooled_mode():
    frame,masks,d=study();d['features']=[{'id':'prop','metric':'proportion','denominator':'tissue'}]
    r=statistics.run(frame,masks,d)
    assert all(s['value']==pytest.approx(100/3) for s in r['samples'])
    d['features'][0]['denominator']='t0'
    with pytest.raises(ValueError,match='contain every cell'):statistics.run(frame,masks,d)
    d['features']=[{'id':'m','column':'M','metric':'mean'}];d['group_by']=''
    r=statistics.run(frame,masks,d)
    assert r['sampled_display'] and all(s['N']==26 for s in r['summaries'])
    d['test']='wilcoxon'
    with pytest.raises(ValueError,match='biological-unit'):statistics.run(frame,masks,d)


def test_dunn_ties_tukey_and_bonferroni_against_independent_formulas():
    arrays=[np.array([1,2,2,4.]),np.array([3,3,4,5.]),np.array([6,7,8,8.])]
    groups=[dict(id=str(i),values=a,units=[]) for i,a in enumerate(arrays)]
    rows=statistics.compare(groups,'kruskal');joined=np.concatenate(arrays);ranks=stats.rankdata(joined)
    assert rows[0]['p']==pytest.approx(stats.kruskal(*arrays).pvalue)
    variance=len(joined)*(len(joined)+1)/12*stats.tiecorrect(ranks)
    for row,(i,j) in zip(rows[1:],[(0,1),(0,2),(1,2)]):
        z=abs(ranks.reshape(3,4).mean(axis=1)[i]-ranks.reshape(3,4).mean(axis=1)[j])/np.sqrt(variance/2)
        assert row['p']==pytest.approx(2*stats.norm.sf(z))
    rows=statistics.compare(groups,'anova');tukey=stats.tukey_hsd(*arrays)
    for row,(i,j) in zip(rows[1:],[(0,1),(0,2),(1,2)]):assert row['tukey_p']==pytest.approx(tukey.pvalue[i,j])
    frame,masks,d=study();d['groups'].append({'id':'g2','population':'t2'});d['test']='anova';d['correction']='bonferroni'
    r=statistics.run(frame,masks,d)
    np.testing.assert_allclose([t['q'] for t in r['tests']],np.minimum(1,np.array([t['p'] for t in r['tests']])*len(r['tests'])))


def test_transformed_features_and_log2_invalid_values():
    frame,masks,d=study()
    axis=flow.fit_axis(frame,{'column':'M','pipeline':[{'op':'arcsinh','cofactor':2},{'op':'zscore'}]})
    d['features']=[{'id':'m','metric':'mean','axis':axis,'column':'M'}]
    r=statistics.run(frame,masks,d)
    x=flow.measurement(frame,axis)
    assert r['samples'][0]['value']==pytest.approx(x[masks['t0'][0]&(frame.patient=='1').to_numpy()].mean())
    d['heatmap']={'mode':'log2','matched':True,'pairs':[{'id':'delta','before':'g0','after':'g1'}]}
    r=statistics.run(frame,masks,d)
    assert r['heatmap'][0]['value'] is None and 'positive' in r['heatmap'][0]['status']


def test_empty_cohort_log2_and_pooled_proportion_have_explicit_status():
    frame,masks,d=study()
    masks['t1']=(np.zeros(len(frame),bool),np.ones(len(frame),bool))
    d['heatmap']={'mode':'log2','matched':False,'pairs':[{'id':'empty','before':'g0','after':'g1'}]}
    result=statistics.run(frame,masks,d)
    assert all(tile['value'] is None and tile['p'] is None for tile in result['heatmap'])
    assert all('No eligible' in tile['status'] for tile in result['heatmap'])
    frame,masks,d=study();d['group_by']='';d['features']=[{'id':'p','metric':'proportion','denominator':'root'}]
    r=statistics.run(frame,masks,d)
    assert r['tests'][0]['p'] is None and 'descriptive' in r['tests'][0]['status']


def test_missing_blank_ids_and_exact_box_whiskers():
    frame,masks,d=study();frame.loc[frame.patient=='1','patient']=' '
    result=statistics.run(frame,masks,d)
    assert all(s['N']==4 for s in result['summaries'])
    assert result['summaries'][0]['missing_units']==3
    summary=statistics.describe([1,2,3,4,100])
    assert summary['upper_whisker']==4 and summary['lower_whisker']==1


def test_population_count_and_category_proportion_keep_zero_patients():
    frame,masks,d=study()
    # The same phenotype occurs in different gate branches and is absent in
    # patient 1. Patient 5 is excluded by the upstream tissue gate.
    frame['phenotype']=np.where((frame.patient.astype(int)>1)&(frame.M%1<.15),'Treg','CD8')
    target={'column':'phenotype','values':['Treg']}
    d['features']=[{'id':'n','metric':'count','target_rule':target},
                   {'id':'p','metric':'proportion','target_rule':target,'denominator':'@group'}]
    d['test']='paired_t'
    d['heatmap']={'mode':'delta','matched':True,'pairs':[{'id':'d','before':'g0','after':'g1'}]}
    r=statistics.run(frame,masks,d)
    assert len(r['samples'])==20
    for group in d['groups']:
        cohort=frame.loc[masks[group['population']][0]]
        den=cohort.groupby('patient').size()
        nums=cohort.phenotype.eq('Treg').groupby(cohort.patient).sum()
        for metric in ('n','p'):
            rows=[s for s in r['samples'] if s['group']==group['id'] and s['feature']==metric]
            for row in rows:
                n,total=int(nums[row['unit']]),int(den[row['unit']])
                assert row['value']==pytest.approx(n if metric=='n' else n/total*100)
                assert row['n_target']==n and row['n_cells']==total
            assert next(s for s in rows if s['unit']=='1')['value']==0
    assert all(t['matched_N']==5 for t in r['tests'])
    d['min_cells']=5
    r=statistics.run(frame,masks,d)
    assert all(s['unit'] not in ('1','2','5') for s in r['samples'])


def test_exact_target_gate_preserves_ancestry_and_cohort_filters():
    frame,masks,d=study()
    d['groups'][0]['rule']={'column':'patient','values':['1','2']}
    d['features']=[{'id':'n','metric':'count','target_population':'t0'},
                   {'id':'p','metric':'proportion','target_population':'t0','denominator':'@group'}]
    r=statistics.run(frame,masks,d)
    assert all(s['value']==0 for s in r['samples'] if s['group']=='g1')
    assert {s['unit'] for s in r['samples'] if s['group']=='g0'}=={'1','2'}
    assert all(s['value']==100 for s in r['samples'] if s['group']=='g0' and s['feature']=='p')
    d['features'][0]['target_population']='deleted'
    with pytest.raises(ValueError,match='target population'):statistics.run(frame,masks,d)


def test_total_counts_pooled_counts_and_proportions_are_descriptive():
    frame,masks,d=study()
    d['features']=[{'id':'n','metric':'count'},
                   {'id':'p','metric':'proportion','target_rule':{'column':'patient','values':['1']},'denominator':'@group'}]
    r=statistics.run(frame,masks,d)
    assert [s['value'] for s in r['samples'] if s['feature']=='n' and s['group']=='g0']==[3,4,5,6,8]
    d['group_by']=''
    r=statistics.run(frame,masks,d)
    assert [s['value'] for s in r['samples']]==pytest.approx([26,26,300/26,300/26])
    assert all(t['p'] is None and t['q'] is None for t in r['tests'])
    assert all(s['N']==1 for s in r['summaries'])
