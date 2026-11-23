import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from spacemap_gui.studio import screening as s, screening_graph as g


def design(**changes):
    c = s.defaults(dict(sample_col="subject", image_col="image", phenotype_col="type", x_col="x", y_col="y",
        distance_col="distance", distance_unit="px", pixel_um=.5, widths=[5, 10], min_cells=1,
        phenotypes=["A", "B", "C"], families=["proportion"], permutations=999,
        endpoint={"type":"binary", "group_col":"outcome", "group_a":"Control", "group_b":"Case"}))
    c.update(changes)
    return c


def cells():
    rows = []
    for sample in range(8):
        for image in range(2):
            for i in range(12):
                rows.append(dict(subject="S"+str(sample), image=f"S{sample}_{image}", x=(i%4)*2, y=(i//4)*2,
                    type=("B" if sample < 4 and i < 6 else ["A","B","C"][i%3]), distance=i*2,
                    outcome="Control" if sample<4 else "Case", label=i+1))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("rings", [False, True])
@pytest.mark.parametrize("triangles", [False, True])
def test_exact_graph_matches_brute_force(rings, triangles):
    rng = np.random.default_rng(37)
    xy = rng.uniform(0, 10, (25,2))
    labels = rng.integers(0, 4, 25)
    zones = rng.integers(-1, 3, 25)
    edges = g.radius_edges(xy, 1, 5)
    got, totals = g.motif_counts(edges, labels, zones, 4, 3, triangles=triangles, rings=rings)
    size = 3 if triangles else 2
    expected = np.zeros_like(got)
    dist = squareform(pdist(xy))
    for zone in range(3):
        for vertices in itertools.combinations(range(len(xy)),size):
            z = zones[list(vertices)]
            if (z < 0).any() or (not (z == zone).all() if rings else (z > zone).any()):
                continue
            if not all(1 <= dist[a,b] <= 5 for a,b in itertools.combinations(vertices,2)):
                continue
            code=0
            for label in sorted(labels[list(vertices)]):
                code=code*4+label
            expected[zone,code]+=1
    np.testing.assert_array_equal(got,expected)
    np.testing.assert_array_equal(totals,expected.sum(axis=1))


def test_exact_expected_frequency_matches_all_label_assignments():
    labels=[0,0,1,1,2]
    permutations=set(itertools.permutations(labels))
    for size in (2,3):
        for motif in itertools.combinations_with_replacement(range(3),size):
            measured=np.mean([tuple(sorted(row[:size]))==motif for row in permutations])
            assert g.label_probability([2,2,1],motif)==pytest.approx(measured)


def test_density_areas_and_all_seven_features(tmp_path):
    frame=cells()
    c=design(families=list(s.FAMILIES), area_table=str(tmp_path/'areas.csv'))
    c['options']['density']['area_source']='csv'
    regions=[]
    for image,p in frame.groupby('image'):
        for zone in range(2):
            regions.append(dict(sample=p.subject.iloc[0],group=p.outcome.iloc[0],image=image,region='All',zone=zone,area_mm2=.1))
    pd.DataFrame(regions).to_csv(c['area_table'],index=False)
    results,samples,features,audit,merged=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    assert set(results.family)==set(s.FAMILIES)
    assert audit['independent_samples']==8
    assert samples.groupby(['feature','zone']).size().max()==8  # 16 images, only eight independent units
    density=features[features.family=='density']
    np.testing.assert_allclose(density.value,density.numerator/.1)
    assert results.q.dropna().ge(results.p.dropna()).all()
    assert len(results)==2*(3+3+3+1+6+3+1)


def test_selection_does_not_change_denominator_and_zero_counts_survive():
    frame=cells()
    frame.loc[(frame.subject=='S0') & (frame.type=='A'),'type']='B'
    c=design(phenotypes=['A'])
    _,_,features,_,_=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    assert (features.loc[features._sample=='S0','value']==0).all()
    assert features.denominator.max()==11
    c['phenotypes']=['A','B','C']
    _,_,all_features,_,_=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    np.testing.assert_allclose(features.value,all_features[all_features.label=='A'].value)


def test_units_boundaries_and_rings():
    c=design(widths=[5,10],zone_mode='rings')
    assert s._zone_index(np.array([-1,0,5,5.01,10,11,np.nan]),c).tolist()==[-1,0,0,1,1,-1,-1]
    c['side']='inside'
    assert s._zone_index(np.array([-10,-5,-.1,0,1]),c).tolist()==[1,0,0,0,-1]


def test_polygon_area_preserves_holes_and_matches_membership(tmp_path):
    from shapely.geometry import box, mapping
    path=tmp_path/'im.geojson'
    study=box(0,0,20,20).difference(box(2,2,4,4))
    target=box(0,0,10,20)
    path.write_text(json.dumps({'features':[{'properties':{'classification':{'name':name}},'geometry':mapping(geom)} for name,geom in [('study',study),('target',target)]]}))
    c=design(zone_source='polygons',annotation_folder=str(tmp_path),study_class='study',target_class='target',widths=[2,5])
    z,area,_,_=s._polygons('im',np.array([[11,5],[14,5],[19,5]])*.5,.5,c)
    assert z.tolist()==[0,0,1]
    assert area[0]==pytest.approx(4*20*.25/1e6)
    c.update(side='inside',widths=[20])
    z,area,_,_=s._polygons('im',np.array([[3,3],[8,5]])*.5,.5,c)
    assert z.tolist()==[-1,0]
    assert area[0]==pytest.approx((200-4)*.25/1e6)


def test_contradictory_sample_groups_rejected_and_paired_handled():
    frame=cells()
    frame.loc[frame.image.str.endswith('_1'),'outcome']='Case'
    with pytest.raises(ValueError,match='both comparison groups'):
        s.compute(frame,design(),{i:.5 for i in frame.image.unique()})
    c=design();c['endpoint']['paired']=True
    results,_,regions,audit,merged=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    assert results.n_a.max()==4
    payload=s._display_result({'values':c,'table':'fixture','source':{},'mpp':{}},Path('fixture'),results,merged,regions,audit,[])
    for hit in payload['hits']:
        assert len(hit['observations'])==2*hit['n_samples']
        assert {p['sample'] for p in hit['observations']}=={'S0','S1','S2','S3'}


def test_independent_groups_difference_detected_and_missing_preserved():
    frame=cells()
    frame.loc[(frame.subject<'S4') & (frame.type=='A'),'type']='B'
    c=design(phenotypes=['A'],widths=[20])
    result,_,_,_,_=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    assert result.iloc[0].effect>0 and result.iloc[0].p<.05
    c['filters']=[{'column':'subject','values':['S4','S5','S6','S7']}]
    result,samples,_,audit,_=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    assert samples.value.isna().sum()==4
    assert result.p.isna().all() and audit['missing_sample_features']==4


def test_validation():
    names=cells().columns
    c=design();c['distance_unit']=''
    with pytest.raises(ValueError,match='distance unit'):s.check(c,names)
    c=design(families=['density'])
    with pytest.raises(ValueError,match='measured zone areas'):s.check(c,names)
    c=design(widths=[100,50])
    with pytest.raises(ValueError,match='increasing'):s.check(c,names)
    with pytest.raises(ValueError,match='too dense|pairs'):
        g.radius_edges(np.zeros((100,2)),0,10,max_edges=5)


def test_saved_job_exports_and_stale_link(tmp_path):
    from spacemap_gui.studio import folders
    frame=cells();table=tmp_path/'cells.parquet';frame.to_parquet(table,index=False)
    folders.change(tmp_path,'tab8','Spatial results','outputs/custom_spatial')
    settings=s.prepare(tmp_path,table,design(),frame.image.unique().tolist())
    result=s.job(settings,None,lambda *a:None)
    out=Path(result['folder'])
    assert out.parent==tmp_path/'outputs/custom_spatial/feature_screening'
    assert len(result['figures'])==1
    assert all((out/'figures'/('proportion.'+ext)).is_file() for ext in ('png','pdf','svg'))
    folders.change(tmp_path,'tab8','Spatial results','outputs/another_spatial')
    assert s.runs(tmp_path,table)[0]['run']==result['run']
    assert s.result(tmp_path,result['run'])['folder']==str(out)
    hit=result['hits'][0]
    found=s.context_cells(tmp_path,table,result['run'],'S0_0',hit['feature'],hit['zone'])
    assert found['column']=='type'
    frame.to_parquet(table,index=False)
    with pytest.raises(ValueError,match='changed'):
        s.context_cells(tmp_path,table,result['run'],'S0_0',hit['feature'],hit['zone'])


def test_live_previews_never_claim_final_q_values():
    f=cells();c=design(families=['proportion','dyad','triad'])
    updates=[]
    def publish(result,merged,regions,audit,message):
        updates.append((result.copy(),message))
        assert result.q.isna().all()
        assert len(merged)>0
    final,_,_,_,_=s.compute(f,c,{i:.5 for i in f.image.unique()},publish=publish)
    assert any('descriptive preview' in m for _,m in updates)
    assert any('Compared' in m for _,m in updates)
    assert final.q.notna().any()


def test_saved_paths_and_pending_labels_revision(tmp_path):
    from spacemap_gui.studio.settings import revision
    chosen=s.defaults({'area_table':str(tmp_path/'areas.csv')})
    initial=s.settings(tmp_path)
    saved=s.save(tmp_path,chosen,initial['revision'])
    assert saved['values']['area_table']==str(tmp_path/'areas.csv')
    assert saved['revision']==revision(chosen)


def test_classic_random_label_scores_are_shared_and_exact():
    from spacemap_gui.tabs.tab8_analysis import feature_screening_core as classic
    rng=np.random.default_rng(173)
    frame=pd.DataFrame({'x':rng.uniform(0,40,60),'y':rng.uniform(0,40,60),
                        'type':rng.choice(['A','B','C'],60), 'distance':rng.uniform(0,30,60),
                        '_image':'image','_sample':'subject','_group':'Control','_region':'All'})
    c=design(families=['dyad','triad'], coordinate_unit='um', distance_unit='um', widths=[15,30], motif_permutations=199, motif_use_gpu=False)
    for f in c['families']:c['options'][f].update(min_um=2,max_um=15,metric='zscore')
    got,audit,_=s._measure(frame,c,1,lambda *a:None)
    assert len([a for a in audit if a['family']=='random-label null'])==2  # shared between families
    for run in [a for a in audit if a['family']=='random-label null']:
        part=frame[frame.distance<=c['widths'][run['zone']]]
        xy=part[['x','y']].to_numpy()
        adj=classic._build_adjacency(xy,2,15)
        edges=classic._build_edge_array(adj)
        tri=classic._find_triangle_array(adj)
        # The bounded numba enumerator gives every Classic triangle exactly once.
        np.testing.assert_array_equal(g.triangle_array(edges,len(part)),tri)
        expected=classic._permutation_zscore_dyad_triad(part.type.to_numpy(),edges,tri,199,
                    np.random.default_rng(run['seed']),min_obs_count=3,use_gpu=False)
        assert run['backend']=='cpu'
        for row in [r for r in got if r['zone']==run['zone']]:
            motif=tuple(sorted(json.loads(row['feature'])[1:]))
            value=expected[int(row['family']=='triad')].get(motif,np.nan)
            np.testing.assert_allclose(row['value'],value,equal_nan=True,rtol=0,atol=0)
    with pytest.raises(ValueError,match='triplets'):
        g.triangle_array(edges,len(part),max_triangles=1)


def test_density_classic_estimate_calibration_and_missing_area():
    from spacemap_gui.tabs.tab8_analysis import feature_screening_core as classic
    frame=cells().query("image == 'S0_0'").copy()
    for k,value in [('_sample','S0'),('_image','S0_0'),('_group','Control'),('_region','All')]:frame[k]=value
    c=design(families=['density'], widths=[30]);c['options']['density']['area_source']='estimate'
    rows,_,_=s._measure(frame,c,.5,lambda *a:None)
    expected,_=classic._compute_virtual_area_mm2(frame[['x','y']].to_numpy(),.5,0,False,95)
    for r in rows:
        assert r['area_mm2']==pytest.approx(expected)
        assert r['value']==pytest.approx(r['numerator']/expected)
    rows,_,_=s._measure(frame.iloc[:1],c,.5,lambda *a:None)
    assert all(np.isnan(r['value']) for r in rows)  # never invent area for one cell
    old=s.defaults({'area_table':'areas.csv','options':{'density':{}}})
    assert old['options']['density']['area_source']=='csv'


def test_binary_mann_whitney_agrees_with_classic():
    from spacemap_gui.tabs.tab8_analysis import feature_screening_core as classic
    frame=cells();c=design(phenotypes=['A'],widths=[30])
    results,_,_,_,merged=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    cf=merged.rename(columns={'_sample':'patient'}).assign(stratum_label='All')
    expected=classic._screen_binary_feature(cf,{'_group_key':'_group','config':{'screening':{'min_patients_per_group':3}}})
    assert results.iloc[0].p==expected['p_value']
    assert results.iloc[0].test=='Mann–Whitney (Classic)'


def test_all_diversity_measures_share_selected_composition_and_keep_missing():
    c=design(families=['diversity'],widths=[20])
    c['options']['diversity']['metrics']=list(s.DIVERSITY)
    frame=pd.DataFrame({'x':range(9),'y':0,'type':['A','B','C','C']+['D']*5,'distance':1,
                        '_sample':'S','_group':'Control','_image':'im','_region':'All'})
    rows,_,_=s._measure(frame,c,.5,lambda *a:None)
    got={json.loads(r['feature'])[1]:r['value'] for r in rows}
    p=np.array([.25,.25,.5]);h=-np.sum(p*np.log(p))
    assert got==pytest.approx(dict(shannon=h,simpson=.625,richness=3,evenness=h/np.log(3)))
    assert all(r['numerator']==4 and r['cells']==9 for r in rows)  # D excluded from diversity, not zone
    frame['type']='A'
    rows,_,_=s._measure(frame,c,.5,lambda *a:None)
    assert rows[0]['value']==0 and rows[1]['value']==0 and rows[2]['value']==1
    assert np.isnan(rows[3]['value']) and 'at least two' in rows[3]['reason']
    frame['type']='D'
    rows,_,_=s._measure(frame,c,.5,lambda *a:None)
    assert all(np.isnan(r['value']) for r in rows)


def test_diversity_selection_legacy_and_validation():
    old=design(families=['diversity']);old['options']['diversity']['metric']='simpson'
    assert s.diversity_metrics(s.defaults(old))==['simpson']
    c=design(families=['diversity']);c['options']['diversity']['metrics']=[]
    with pytest.raises(ValueError,match='at least one diversity'):s.check(c,cells().columns)
    c['families']=['proportion'];s.check(c,cells().columns)  # unselected family may be empty
    for value in [['simpson','simpson'],['bad'],'shannon',None]:
        c['options']['diversity']['metrics']=value
        with pytest.raises(ValueError,match='diversity measures'):s.check(c,cells().columns)


def test_all_diversity_tests_enter_global_correction_and_separate_exports(tmp_path):
    from spacemap_gui.studio import screening_figures as figures
    from statsmodels.stats.multitest import multipletests
    frame=cells();c=design(families=['diversity','proportion'], widths=[5,10])
    c['options']['diversity']['metrics']=list(s.DIVERSITY)
    results,_,regions,audit,merged=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    assert len(results)==14  # four diversity + three proportions, two zones
    expected=multipletests(results.p.fillna(1),method=c['correction'])[1]
    expected[results.p.isna()]=np.nan
    np.testing.assert_allclose(results.q,expected)
    paths=figures.save(tmp_path,results,merged,c)
    div=[p for p in paths if p['family']=='diversity']
    assert {p['metric'] for p in div}==set(s.DIVERSITY)
    for p in div:
        svg=(tmp_path/'figures'/p['file'].replace('.png','.svg')).read_text()
        assert 'Diversity · '+s.DIVERSITY[p['metric']] in svg
        assert 'all 14 planned tests' in svg
        for other in set(s.DIVERSITY)-{p['metric']}:
            assert 'Diversity · '+s.DIVERSITY[other] not in svg


def test_diversity_preview_does_not_hide_other_measures():
    f=cells();c=design(families=['diversity'],widths=list(range(1,31)))
    c['options']['diversity']['metrics']=list(s.DIVERSITY)
    result,_,regions,audit,merged=s.compute(f,c,{i:.5 for i in f.image.unique()})
    payload=s._display_result({'values':c,'table':'fixture','source':{},'mpp':{}},Path('fixture'),result,merged,regions,audit,[])
    assert len(payload['hits'])==120
    assert all(r['phenotypes']==c['phenotypes'] for r in payload['hits'])


def test_density_missing_tumor_zone_is_missing_but_absent_cell_type_is_zero():
    frame=cells()
    frame.loc[frame.subject=='S0','distance']=np.nan  # no tumor distance in either image
    frame.loc[frame.image=='S1_0','distance']=np.nan  # one image remains usable
    frame.loc[(frame.subject=='S2') & (frame.type=='A'),'type']='B'  # measurable zone, no A
    c=design(families=['density'],phenotypes=['A'],widths=[30])
    c['options']['density']['area_source']='estimate'
    _,samples,regions,_,_=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    assert regions.loc[regions._sample=='S0','value'].isna().all()
    assert np.isnan(samples.set_index('_sample').loc['S0','value'])
    assert samples.set_index('_sample').loc['S1','n_regions']==1
    assert samples.set_index('_sample').loc['S2','value']==0


def test_statistics_export_and_estimated_density_are_explicit(tmp_path):
    from spacemap_gui.studio import screening_figures as figures
    frame=cells();c=design(phenotypes=['A'],widths=[30])
    result,_,_,_,merged=s.compute(frame,c,{i:.5 for i in frame.image.unique()})
    text=figures.statistics_text(result.iloc[0],c)
    assert 'p =' in text and 'q =' in text and '8 samples' in text and 'Mann–Whitney' in text
    figures.save(tmp_path,result,merged,c)
    svg=(tmp_path/'figures/proportion.svg').read_text()
    assert 'p =' in svg and 'q =' in svg and 'n=4' in svg
    c['options']['density']['area_source']='estimate'
    assert 'estimated' in figures.measurement(c,'density')


def test_numeric_and_survival_at_sample_level():
    rng=np.random.default_rng(19)
    n=40
    values=rng.normal(size=n)
    samples=pd.DataFrame({'_sample':[f'S{i}' for i in range(n)],'_group':'All','family':'proportion',
                          'feature':'["proportion", "A"]','label':'A','zone':0,'value':values})
    outcomes=samples[['_sample','_group']].copy()
    outcomes['numeric']=values+rng.normal(0,.1,n)
    c=design();c['endpoint'].update(type='continuous',value_col='numeric')
    result,_=s.infer(samples,outcomes,c)
    assert result.iloc[0].effect>.95 and result.iloc[0].p<.01
    outcomes['time']=rng.exponential(20*np.exp(-values))+.01
    outcomes['event']=rng.integers(0,2,n)
    c['endpoint'].update(type='survival',time_col='time',event_col='event')
    result,_=s.infer(samples,outcomes,c)
    assert result.iloc[0].n_samples==n
    assert result.iloc[0]['test']=='Cox HR per SD'
    assert result.iloc[0].effect>1


def test_classic_dyad_count_is_row_order_invariant():
    from spacemap_gui.tabs.tab8_analysis import feature_screening_core as core
    xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
    adjacency=core._build_adjacency(xy,0,2)
    assert core._count_dyads(np.array(['B','A','B']),adjacency,'A','B')==2
    assert core._count_dyads(np.array(['A','B','B']),adjacency,'A','B')==2
