"""Crosshair readouts preserve source intensities and share native tile reads."""
import numpy as np
import tifffile

from spacemap.viewer import render
from spacemap.viewer.sources import open_source
from spacemap_gui.viewer import registry
from test_viewer import client, _write_channel_pages  # noqa: F401


def test_native_pixel_values_batched_cached_and_at_edges(client, tmp_path, monkeypatch):
    path = tmp_path / 'pixels.ome.tiff'
    _write_channel_pages(path, channels=3, width=600, height=600)
    token = registry.register(path)
    info = open_source(path).info
    render.raw_tile(str(path), info, 0, 1, 1, 0)
    render.raw_tile(str(path), info, 0, 1, 1, 2)
    # Once rendered, numerical readouts reuse those raw pixels without disk I/O.
    monkeypatch.setattr(render, 'open_source', lambda _: (_ for _ in ()).throw(AssertionError('Unexpected image read')))
    result = client.get(f'/image/{token}/pixels', params={'x':512.9,'y':513.2,'ch':'2,0,2'})
    expected = (513*600+512)%4000
    assert result.status_code == 200
    assert result.json() == {'x':512,'y':513,'values':{'2':expected+2000,'0':expected}}
    edge = client.get(f'/image/{token}/pixels?x=599.99&y=599.99&ch=0').json()
    assert edge['values']['0'] == (599*600+599)%4000
    for x, y in [(-.01,0),(600,0),(0,-1),(0,600)]:
        assert client.get(f'/image/{token}/pixels',params={'x':x,'y':y,'ch':'0,2'}).json()['values']=={'0':None,'2':None}


def test_pixel_values_keep_zero_signed_floats_and_missing(client, tmp_path):
    path = tmp_path / 'float.tif'
    data = np.zeros((16,16),dtype=np.float32)
    data[0,:5] = [-2.75,0,1.125,np.nan,np.inf]
    tifffile.imwrite(path,data,photometric='minisblack',metadata={'axes':'YX'})
    token = registry.register(path)
    for x, expected in enumerate([-2.75,0,1.125,None,None]):
        result=client.get(f'/image/{token}/pixels?x={x}&y=0&ch=0')
        assert result.status_code == 200 and result.json()['values']['0']==expected


def test_pixel_readout_validates_requests(client, tmp_path):
    path = tmp_path / 'pixels.ome.tiff'
    _write_channel_pages(path, channels=2, width=64, height=64)
    token = registry.register(path)
    for query in ('x=nan&y=0&ch=0','x=0&y=inf&ch=0','x=0&y=0&ch=-1',
                  'x=0&y=0&ch=2','x=0&y=0&ch=a','x=0&y=0','y=0&ch=0'):
        assert client.get(f'/image/{token}/pixels?{query}').status_code==400
    assert client.get('/image/unknown/pixels?x=0&y=0&ch=0').status_code==404
