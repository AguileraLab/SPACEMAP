import copy
import numpy as np
import pandas as pd
import pytest
from spacemap import flow_gating as gates, flow_transforms as recipe


def test_order_skip_and_named_steps_are_respected():
    frame=pd.DataFrame({'a':[1.,2,8,100],'b':[3.,5,1,10]})
    spec={'column':'a','pipeline':[{'op':'zscore','name':'Across cells'},{'op':'clip','low':10,'high':90},
                                 {'op':'minmax'},{'op':'log10','enabled':False}]}
    fitted=recipe.fit(frame,spec)
    result=recipe.transform(frame,fitted)
    expected=frame.a.to_numpy();expected=(expected-expected.mean())/expected.std()
    expected=np.clip(expected,*np.percentile(expected,[10,90]));expected=(expected-expected.min())/(expected.max()-expected.min())
    assert np.allclose(result,expected)
    assert fitted['pipeline'][0]['name']=='Across cells'
    # Applying a frozen recipe to descendants never refits their range.
    assert np.allclose(recipe.transform(frame.iloc[:2],fitted),expected[:2])
    assert 'fitted' not in spec['pipeline'][0]


def test_z1_z2_cdf_nlog_match_studio_values():
    from spacemap.viewer.marker_transforms import transform
    frame=pd.DataFrame({'a':[1.,2,4,8],'b':[2.,1,8,4],'c':[4.,7,2,1]})
    ops=[{'op':'arcsinh','cofactor':5},{'op':'zscore'},{'op':'row_zscore'},{'op':'cdf'},{'op':'nlog'}]
    spec={'column':'a','markers':['a','b','c'],'pipeline':ops}
    result=recipe.transform(frame,recipe.fit(frame,spec))
    raw=frame.to_numpy();a=np.arcsinh(raw/5)
    stats=(np.stack([raw.mean(0),a.mean(0)]),np.stack([raw.std(0),a.std(0)]),np.full(3,4))
    expected=np.array([transform(row,['Arcsinh','Z1','Z2','CDF','NLog'],stats)[0] for row in raw])
    assert np.allclose(result,expected)


def test_dynamic_cofactor_and_fixed_reference_are_reproducible():
    frame=pd.DataFrame({'a':[1.,4,10,100],'slide':[1,1,2,2]})
    spec={'column':'a','pipeline':[{'op':'arcsinh','dynamic':True,'cofactor_percentile':50}]}
    fitted=recipe.fit(frame,spec,np.array([True,True,False,False]))
    assert fitted['pipeline'][0]['fitted']['all']['a']['cofactor']==2.5
    assert np.allclose(recipe.transform(frame,fitted),np.arcsinh(frame.a/2.5))
    zero=pd.DataFrame({'a':[0.,0]})
    with pytest.raises(ValueError,match='cofactor is zero'):
        recipe.fit(zero,spec)


def test_missing_groups_constant_columns_and_nonpositive_values():
    frame=pd.DataFrame({'a':[3.,3,0,-2,np.nan],'slide':['x','x','y','y',None]})
    spec={'column':'a','pipeline':[{'op':'zscore','group':'slide'}]}
    result=recipe.transform(frame,recipe.fit(frame,spec))
    assert np.allclose(result[:4],[0,0,1,-1]) and np.isnan(result[4])
    spec['pipeline']=[{'op':'log1p'}]
    result=recipe.transform(frame,recipe.fit(frame,spec))
    assert np.isnan(result[3:]).all() and result[2]==0


def test_recipe_gate_coordinates_stay_fixed_after_plot_changes():
    frame=pd.DataFrame({'a':[1.,2,3,4],'b':[4.,3,2,1]})
    axis=gates.fit_axis(frame,{'column':'a','pipeline':[{'op':'arcsinh','cofactor':2},{'op':'minmax'}]})
    gate={'id':'a','kind':'range','x':axis,'bounds':[.4,1]}
    before=gates.evaluate(frame,[gate])['a'][0].copy()
    another=copy.deepcopy(axis);another['pipeline'].reverse();another.pop('pipeline_fitted')
    gates.fit_axis(frame,another)
    assert np.array_equal(before,gates.evaluate(frame,[gate])['a'][0])


def test_real_logicle_roundtrip_when_optional_dependency_present():
    pytest.importorskip('flowutils')
    values=np.array([-1000.,-10,0,1,100,100000])
    spec={'column':'a','scale':'logicle','t':262144,'m':4.5,'w':.5,'a':0}
    assert np.allclose(gates.axis_values(gates.axis_values(values,spec),spec,inverse=True),values,atol=1e-8)


def test_z2_does_not_silently_expand_the_explicit_marker_set():
    frame=pd.DataFrame({'a':[1.,2],'b':[2.,3],'c':[4.,6]})
    with pytest.raises(ValueError,match='explicit Z2 marker set'):
        gates.fit_axis(frame,{'column':'c','markers':['a','b'],'pipeline':[{'op':'row_zscore'}]})


def test_each_z2_step_uses_its_own_markers_and_keeps_other_values():
    frame=pd.DataFrame({'a':[1.,5,9],'b':[3.,2,7],'c':[4.,-3,12]})
    original=frame.copy()
    spec={'column':'a','pipeline':[{'op':'row_zscore','markers':['a','b']},
                                  {'op':'row_zscore','markers':['a','c']}]}
    fitted=recipe.fit(frame,spec)
    assert np.allclose(recipe.transform(frame,fitted),[-1,1,-1])
    assert np.allclose(recipe.transform(frame.copy(),fitted),[-1,1,-1])
    pd.testing.assert_frame_equal(frame,original)
    assert 'markers' not in spec  # the input recipe stays reusable
    spec['pipeline'][1]['markers']=['a','b']
    assert np.allclose(recipe.transform(frame,recipe.fit(frame,spec)),[-1,1,1])


def test_legacy_z2_scope_is_preserved_alongside_new_step_markers():
    frame=pd.DataFrame({'a':[1.,5,9],'b':[3.,2,7],'c':[4.,-3,12]})
    spec={'column':'a','markers':['a','b'],'pipeline':[{'op':'row_zscore'},
          {'op':'row_zscore','markers':['a','c']}]}
    fitted=recipe.fit(frame,spec)
    assert fitted['pipeline'][0]['markers']==['a','b']
    assert np.allclose(recipe.transform(frame,fitted),[-1,1,-1])
    with pytest.raises(ValueError,match='explicit Z2 marker set'):
        recipe.fit(frame,{'column':'a','pipeline':[{'op':'row_zscore','markers':['b','c']}]})
    with pytest.raises(ValueError,match='at least two'):
        recipe.fit(frame,{'column':'a','pipeline':[{'op':'row_zscore','markers':['a']}]})
