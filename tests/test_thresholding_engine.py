"""Threshold parity, raw-unit invariants and benchmark API validation."""
import io
import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import Polygon
from spacemap import thresholding_engine as engine
from spacemap.background_detection import find_background
from spacemap.viewer import render, transforms, display
from spacemap.viewer.sources import open_source
from spacemap_gui.viewer import registry
from spacemap_gui.tabs.tab7_phenotyping.common import compute_auto_thresholds
from test_viewer import _write_channel_pages, client  # noqa: F401


@pytest.mark.parametrize('method',engine.THRESHOLD_METHODS)
def test_background_and_cell_gates_are_identical_for_same_population(method):
    rng=np.random.default_rng(12)
    values=np.r_[rng.normal(100,8,4000),rng.normal(500,20,2000)].astype(np.uint16)
    params={'manual':125.5,'percentile':83}
    expected=engine.calculate(values,method,params=params)['threshold']
    assert find_background(values,method,params=params)['threshold']==expected
    gates=compute_auto_thresholds(pd.DataFrame({'marker':values}),['marker'],method,params=params)
    assert gates.marker.iloc[0]==expected


@pytest.mark.parametrize('method',['Otsu','Geometric elbow','Yen','Percentile'])
def test_corrected_values_are_not_clipped_or_retransformed(method):
    values=np.linspace(-2,4,500)
    expected=engine.calculate(values,method)['threshold']
    assert -2 <= expected <= 4
    for steps in ([],['Arcsinh'],['Log10']):
        transform=transforms.Transform.fit(values,steps)
        assert transforms.automatic_threshold(values,transform,method)['raw']==expected


def test_log_coordinates_are_exact_and_empty_failures_cannot_apply():
    assert engine.project(100,'Log10')==2
    assert np.isnan(engine.project(0,'Log10'))
    assert engine.project(100,'Arcsinh',150)==pytest.approx(np.arcsinh(100/150))
    for values in ([],[np.nan,np.inf]):
        with pytest.raises(ValueError,match='No finite'):engine.calculate(values)
    with pytest.raises(ValueError,match='between'):engine.calculate([1,2],'Percentile',params={'percentile':101})
    for method in engine.THRESHOLD_METHODS:
        if method!='Manual':assert engine.calculate(np.full(10,-1.25),method)['threshold']==-1.25


def test_benchmark_mask_holes_and_display_scale_invariant(client,tmp_path):
    path=tmp_path/'image.ome.tiff';_write_channel_pages(path,channels=1,height=120,width=160)
    info=open_source(path).info;token=registry.register(path)
    shape=Polygon([(10,10),(90,10),(90,100),(10,100)],holes=[[(30,30),(50,30),(50,50),(30,50)]])
    from shapely.geometry import mapping
    annotation=tmp_path/'mask.geojson';annotation.write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','geometry':mapping(shape),'properties':{'classification':{'name':'tissue'}}}]}))
    ann=registry.register_object('annotations',path=str(annotation),image=str(path))
    q={'ch':0,'methods':'Manual,Percentile,Otsu','manual':100,'percentile':90,'annotation':ann,'class':'tissue','native':'1'}
    payload=[]
    for scale in ['Linear','Log10','Arcsinh']:
        response=client.get(f'/image/{token}/threshold_benchmark',params={**q,'scale':scale,'cofactor':150})
        assert response.status_code==200,response.text
        data=response.json();payload.append(data)
        assert len(data['populations'])==2
        whole,mask=data['populations'];assert mask['pixels']<whole['pixels']==120*160
        vals,_=render.marker_pixels(str(path),info,0,region=[shape],native=True)
        assert mask['pixels']==len(vals)
        assert sum(mask['counts'])+mask['omitted']==mask['pixels']
        assert all(t['threshold']==engine.calculate(vals,t['method'],params=q)['threshold'] for t in mask['thresholds'])
    assert payload[1]['populations'][0]['thresholds'][0]['coordinate']==2
    for pi in range(2):
        assert [t['threshold'] for t in payload[0]['populations'][pi]['thresholds']]==[t['threshold'] for t in payload[2]['populations'][pi]['thresholds']]
    q['class']='missing';assert client.get(f'/image/{token}/threshold_benchmark',params=q).status_code==400
    q.pop('class');q['annotation']=registry.register_object('annotations',path=str(annotation),image=str(tmp_path/'other.tiff'))
    assert client.get(f'/image/{token}/threshold_benchmark',params=q).status_code==400


def test_binary_preview_matches_raw_cutoff_and_invalid_parameters_fail(client,tmp_path):
    path=tmp_path/'image.ome.tiff';_write_channel_pages(path,channels=1,height=80,width=100)
    token=registry.register(path);info=open_source(path).info
    response=client.get(f'/image/{token}/tile/0/0/0.png',params={'threshold_preview':'0:100:ff0000:mask'})
    assert response.status_code==200,response.text
    pixels=np.asarray(Image.open(io.BytesIO(response.content)))
    raw=render.raw_tile(str(path),info,0,0,0,0)
    np.testing.assert_array_equal(pixels[:,:,0]>0,raw>100)
    assert not pixels[:,:,1:].any()
    for value in ['0:nan:ff0000:mask','99:100:ff0000:mask','0:100:bad:mask']:
        assert client.get(f'/image/{token}/tile/0/0/0.png',params={'threshold_preview':value}).status_code==400


def test_parameters_survive_display_reload_and_old_writers(tmp_path):
    path=tmp_path/'display.json';options={'method':'Percentile','scope':'','parameters':{'percentile':87.5,'manual':123},'native':True}
    display.save(path,[{'index':0,'name':'M0','low':123,'high':200}],options)
    display.save(path,[{'index':0,'name':'M0','low':123,'high':200}])
    assert display.load(path)['auto']==options


def test_scale_changes_reuse_samples_and_cutoffs_and_failed_methods_are_explicit(tmp_path,monkeypatch):
    path=tmp_path/'cached.ome.tiff';_write_channel_pages(path,channels=1,height=80,width=100)
    info=open_source(path).info
    calculate=engine.calculate; calls=[]
    def counted(values,method,**kwargs):
        calls.append(method)
        if method=='Minimum':raise ValueError('No two modes in this sample')
        return calculate(values,method,**kwargs)
    monkeypatch.setattr(engine,'calculate',counted)
    first=render.threshold_benchmark(str(path),info,0,['Otsu','Minimum'])
    assert 'error' in first['populations'][0]['thresholds'][1]
    def no_read(*args,**kwargs):raise AssertionError('Display scale must reuse cached raw pixels')
    monkeypatch.setattr(render,'marker_pixels',no_read)
    changed=render.threshold_benchmark(str(path),info,0,['Otsu'],scale='Log10')
    assert calls.count('Otsu')==1
    assert changed['populations'][0]['thresholds'][0]['threshold']==first['populations'][0]['thresholds'][0]['threshold']


def test_studio_export_and_batch_state_preserve_parameters_and_cutoff_precision(tmp_path):
    from spacemap_gui.studio import background
    from spacemap.viewer.display import load
    path=tmp_path/'image.ome.tiff';_write_channel_pages(path,channels=1,height=80,width=100)
    value=123.456789123456
    background.detect([{'image':str(path),'root':str(tmp_path),'method':'Manual',
                        'parameters':{'manual':value,'percentile':88},'native':True}],lambda *args:None)
    levels=background.stored_levels(tmp_path,path)
    assert list(levels.values())==[value]
    assert load(background.display_path(tmp_path,path))['auto']['parameters']=={'manual':value,'percentile':88}
    target=tmp_path/'thresholds.csv';background.save_levels(target,levels,'Manual')
    assert list(background.read_levels(target).values())==pytest.approx([value],abs=1e-12,rel=0)


@pytest.mark.parametrize('method',engine.THRESHOLD_METHODS)
@pytest.mark.parametrize('dtype',[np.uint16,np.int16])
def test_lossless_histograms_match_expanded_pixels_for_every_method(method,dtype):
    rng=np.random.default_rng(31)
    data=np.r_[rng.normal(120,18,5000),rng.normal(500,50,1500)].astype(dtype)
    if dtype==np.int16:data=data-200
    population=render._compact_population(render._integer_counts(data,np.dtype(dtype)),np.dtype(dtype))
    params={'percentile':87.25,'manual':17.125}
    dense=engine.calculate(data,method,params=params)
    compact=engine.calculate(population,method,params=params)
    assert compact['threshold']==pytest.approx(dense['threshold'],rel=0,abs=1e-10)
    assert compact['n_pixels']==dense['n_pixels']==len(data)


def test_native_streaming_counts_cross_tile_polygons_and_holes_exactly(tmp_path):
    path=tmp_path/'stream.ome.tiff';_write_channel_pages(path,channels=1,height=1300,width=1800)
    info=open_source(path).info
    shape=Polygon([(10,10),(1700,20),(1650,1200),(20,1250)],holes=[[(980,950),(1090,950),(1090,1100),(980,1100)]])
    for regions in (None,[shape]):
        raw,_=render.marker_pixels(str(path),info,0,region=regions,native=True)
        compact=render._native_population(str(path),info,0,regions)
        assert compact.size==raw.size
        expected=render._integer_counts(raw,np.dtype(info.dtype))
        actual=render._integer_counts(compact.values,np.dtype(info.dtype)) # check support below
        assert np.array_equal(np.flatnonzero(actual),np.flatnonzero(expected))
        np.testing.assert_array_equal(compact.counts,expected[expected>0])
        result=render.threshold_benchmark(str(path),info,0,['Manual'],params={'manual':100},region=regions,native=True,scale='Log10')
        selected=result['populations'][-1]
        assert selected['thresholds'][0]['positive_fraction']==pytest.approx(np.mean(raw>100))
        assert selected['pixels']==raw.size and selected['downsample']==1
        assert sum(selected['counts'])+selected['omitted']==raw.size


@pytest.mark.parametrize('scale',['Linear','Log10','Arcsinh'])
@pytest.mark.parametrize('native',[False,True])
def test_image_histogram_and_benchmark_share_bins_counts_and_full_range(client,tmp_path,scale,native):
    from shapely.geometry import box
    path=tmp_path/'parity.ome.tiff';_write_channel_pages(path,channels=1,height=100,width=120)
    info=open_source(path).info;region=[box(10,20,70,90)]
    options=dict(scale=scale,cofactor=150,native=native,region=region)
    benchmark=render.threshold_benchmark(str(path),info,0,['Manual'],params={'manual':1e9},**options)
    image=render.channel_histogram(str(path),info,0,**options)
    mask=benchmark['populations'][1]
    assert image['counts']==mask['counts']
    assert image['axis']==benchmark['axis'] and image['edges']==benchmark['edges']
    assert len(image['counts'])==256
    assert sum(image['counts'])+image['omitted']==image['pixels']
    assert benchmark['axis'][1]<1e9  # an off-axis cutoff never changes binning
    whole=render.channel_histogram(str(path),info,0,scale=scale,cofactor=150,native=native)
    assert whole['counts']==benchmark['populations'][0]['counts']
    token=registry.register(path)
    response=client.get(f'/image/{token}/histogram',params={'ch':0,'scale':scale,'cofactor':150,'native':int(native)})
    assert response.status_code==200 and response.json()['counts']==whole['counts']


@pytest.mark.parametrize('low',[-20,0,125])
def test_upper_display_percentile_uses_only_pixels_above_low_without_mutation(tmp_path,low):
    path=tmp_path/'upper.ome.tiff';_write_channel_pages(path,channels=1,height=120,width=140)
    info=open_source(path).info;values,_=render.marker_pixels(str(path),info,0,native=True)
    original=path.stat().st_mtime_ns
    result=render.auto_display_range(str(path),info,0,mode='upper',low=low,native=True)
    assert result['high']==pytest.approx(np.percentile(values[values>low],99.8))
    assert result['low']==low and result['display_only_upper']
    assert result['signal_pixels']==np.count_nonzero(values>low)
    assert path.stat().st_mtime_ns==original
    both=render.auto_display_range(str(path),info,0,mode='both',method='Manual',params={'manual':low},native=True)
    assert both['high']==result['high'] and both['low']==low
    with pytest.raises(ValueError,match='No pixels above'):
        render.auto_display_range(str(path),info,0,mode='upper',low=float(values.max()),native=True)


def test_display_range_route_and_preferences(client,tmp_path):
    path=tmp_path/'upper.ome.tiff';_write_channel_pages(path,channels=1,height=80,width=100)
    token=registry.register(path)
    url=f'/image/{token}/display_range'
    response=client.get(url,params={'ch':0,'mode':'both','method':'Manual','manual':100,'upper_percentile':99.8})
    assert response.status_code==200 and response.json()['low']==100
    assert response.json()['high']>100
    for options in ({'upper_percentile':101},{'upper_percentile':'nan'},{'mode':'invalid'},{'low':'inf','mode':'upper'}):
        assert client.get(url,params={'ch':0,**options}).status_code==400
    prefs={'method':'Manual','display':{'adjust':'upper','upper_percentile':99.5},'histogram':{'scale':'Arcsinh','y':'fraction','cofactor':150}}
    saved=tmp_path/'display.json';display.save(saved,[{'index':0,'name':'M0','low':100,'high':200}],prefs)
    reloaded=display.load(saved)
    assert reloaded['auto']['display']==prefs['display']
    assert reloaded['auto']['histogram']==prefs['histogram']
    assert reloaded['channels'][0]['low']==100
