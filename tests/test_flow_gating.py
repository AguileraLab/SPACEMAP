import numpy as np
import pandas as pd
import pytest

from spacemap import flow_gating as flow


def axis(column='signal', **kw):
    return {'column':column,'scale':'linear',**kw}


def test_boundaries_missing_nested_boolean_and_cycles():
    frame = pd.DataFrame({'signal':[-1,0,1,2,np.nan,np.inf], 'y':[0,0,1,2,0,0]})
    gates = [dict(id='r',parent='root',kind='range',x=axis(),bounds=[0,1]),
             dict(id='n',parent='root',kind='boolean',op='not',refs=['r']),
             dict(id='p',parent='r',kind='polygon',x=axis(),y=axis('y'),vertices=[[0,0],[1,0],[1,1],[0,1]])]
    result = flow.evaluate(frame,gates)
    assert np.flatnonzero(result['r'][0]).tolist() == [1,2]
    assert np.flatnonzero(result['n'][0]).tolist() == [0,3]
    assert np.flatnonzero(result['p'][0]).tolist() == [1,2]
    gates[0]['parent']='p'
    with pytest.raises(ValueError,match='cycle'):
        flow.evaluate(frame,gates)


def test_quadrants_partition_valid_boundary_events():
    frame = pd.DataFrame({'signal':[-1,0,1,0,np.nan], 'y':[-1,0,-1,1,1]})
    gates = [dict(id=str(i),kind='quadrant',x=axis(),y=axis('y'),bounds=[0,0],right=i%2==0,above=i<2) for i in range(4)]
    results = flow.evaluate(frame,gates)
    assert np.stack([results[str(i)][0] for i in range(4)]).sum(axis=0).tolist() == [1,1,1,1,0]


def test_three_valued_filters_and_group_normalization():
    frame = pd.DataFrame({'signal':[1.,3.,9.,9.,4.], 'batch':['a','a','b','b',None]})
    fitted = flow.fit_axis(frame,axis(normalize='zscore',group='batch'))
    assert np.allclose(flow.measurement(frame,fitted)[:4],[-1,1,0,0])
    assert np.isnan(flow.measurement(frame,fitted)[4])
    assert fitted['fitted']['a']['scale'] == 1 # ddof=0 consistently
    result = flow.filter_rule(frame,{'join':'not','rules':[{'column':'batch','op':'in','values':['a']}]})
    assert np.flatnonzero(result[0]).tolist() == [2,3]
    assert not result[1][4]


@pytest.mark.parametrize('scale',['linear','arcsinh','log10'])
def test_transform_roundtrip(scale):
    x = np.array([.01,1,4,100,10000.])
    assert np.allclose(flow.axis_values(flow.axis_values(x,axis(scale=scale)),axis(scale=scale),inverse=True),x)
    if scale=='log10':
        assert np.isnan(flow.axis_values([-1,0],axis(scale=scale))).all()


def test_overlay_shared_bins_exact_counts_and_empty_live():
    frame = pd.DataFrame({'signal':[0,1,2,3,4,np.nan], 'arm':['a']*3+['b']*3})
    results = flow.evaluate(frame,[])
    spec = {'x':axis(),'bins':4,'frequency':'percent','series':[
        {'population':'root','rule':{'column':'arm','values':['a']}},
        {'population':'root','rule':{'column':'arm','values':['b']},'pinned':True}]}
    result = flow.distributions(frame,results,spec,np.zeros(6,bool))
    assert len(result['edges'])==5
    assert result['series'][0]['n']==0
    assert result['series'][1]['invalid']==1
    assert sum(result['series'][1]['counts'])==2
    assert sum(result['series'][1]['values'])==100


def test_histogram_proportions_keep_missing_and_outside_denominators():
    frame = pd.DataFrame({'signal':[0,1,2,3,4,np.nan], 'arm':['a']*3+['b']*3})
    results = flow.evaluate(frame,[])
    spec = {'x':axis(),'bins':4,'frequency':'proportion','x_range':[0,2]}
    series = flow.distributions(frame,results,spec)['series'][0]
    assert series['invalid']==1 and series['outside']==2
    assert np.allclose(series['values'],np.array(series['counts'])/5)
    assert sum(series['values'])==pytest.approx(.6)
    weighted = flow.distributions(frame,results,{**spec,'patient':'arm'})['series'][0]
    assert sum(weighted['values'])==pytest.approx(.5)  # equal weight per arm
    empty = flow.distributions(frame,results,spec,np.zeros(6,bool))['series'][0]
    assert empty['values']==[0.]*4


def test_frame_caches_do_not_copy_large_arrays_or_leak_to_subsets():
    class NoCopy(np.ndarray):
        def __deepcopy__(self,memo):
            raise AssertionError('Pandas must not copy analysis arrays during column operations')
    frame=pd.DataFrame({'signal':[1.,2.,3.,4.],'arm':['a','b',None,'a']})
    frame.attrs['flow_scope']='immutable-test'
    cache=flow.frame_cache(frame)
    cache['large']=np.ones(100).view(NoCopy)
    rule={'column':'arm','op':'in','values':['a']}
    assert flow.filter_rule(frame,rule)[0].tolist()==[True,False,False,True]
    assert frame.arm.dropna().tolist()==['a','b','a']
    subset=frame.iloc[1:3].copy()
    assert flow.filter_rule(subset,rule)[0].tolist()==[False,False]
    assert flow.frame_cache(subset) is not cache
    joined=pd.concat([frame,subset],ignore_index=True)
    assert flow.filter_rule(joined,rule)[0].tolist()==[True,False,False,True,False,False]


@pytest.mark.parametrize('values',[[0.,1.,1.25,np.nan,np.inf],pd.Series([0,1,None,0,1],dtype='Int64'),[True,False,True,False,True]])
def test_encoded_numeric_categories_preserve_text_filter_semantics(values):
    frame=pd.DataFrame({'value':values});frame.attrs['flow_scope']='numeric-test'
    for value in frame.value.astype('string').dropna().unique():
        mask,known=flow.filter_rule(frame,{'column':'value','op':'in','values':[value]})
        assert mask.tolist()==frame.value.astype('string').eq(value).fillna(False).tolist()
        assert known.tolist()==frame.value.notna().tolist()


def test_non_linear_boundary_is_densely_reprojected_without_changing_membership():
    frame=pd.DataFrame({'signal':[1,2,10],'y':[1,3,10]})
    gate=dict(id='p',kind='polygon',x=axis(),y=axis('y'),vertices=[[1,1],[10,1],[10,10]])
    before=flow.evaluate(frame,[gate])['p'][0].copy()
    boundary=flow.reproject_boundary(gate,axis(scale='arcsinh'),axis('y',scale='arcsinh'))
    assert len(boundary)>300
    assert np.array_equal(before,flow.evaluate(frame,[gate])['p'][0])
