"""Restart-safe summaries must track table/journal edits without rescanning cells."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pandas as pd

from spacemap import dataio, labels, pending_columns
from spacemap_gui.studio import phenotyping_overview as overview, summary_cache
from test_studio_commands import setup  # noqa: F401


def test_restart_cache_and_label_grouping_changes(tmp_path, monkeypatch):
    monkeypatch.setenv('SPACEMAP_CACHE_DIR', str(tmp_path/'cache'))
    table = tmp_path/'cells.parquet'
    frame = pd.DataFrame({'image':['A','A','B','B','B'], 'slide':['one']*5,
                          'cell_key':range(5), 'custom':['T',' ',None,'B','T'], 'numeric':[0,1,None,2,3]})
    frame.to_parquet(table,index=False)
    def counts():
        return overview.results(table, suffix='nonexistent', roi_column='image', extra_columns=['custom','numeric'])
    assert counts() == {'custom':{'A':1,'B':2}, 'numeric':{'A':2,'B':2}}
    first = overview.overview(table,'image','slide')
    summary_cache._MEMORY.clear()  # fresh process: disk cache only
    original = dataio.load
    monkeypatch.setattr(dataio,'load',lambda *a,**k: (_ for _ in ()).throw(AssertionError('No cell scan on restart')))
    assert overview.overview(table,'image','slide') == first
    assert counts()['custom']['B'] == 2
    monkeypatch.setattr(dataio,'load',original)
    labels.append(table,[{'row':1,'column':'custom','value':'Treg'},{'row':0,'column':'image','value':'B'}])
    assert counts()['custom'] == {'A':1,'B':3}
    assert {r['roi']:r['cells'] for r in overview.overview(table,'image','slide')['rois']} == {'A':1,'B':4}
    labels.clear(table)
    assert counts()['custom'] == {'A':1,'B':2}
    pending_columns.save(table,{'custom':['X']*5},anchor_column='cell_key',anchor_values=frame.cell_key)
    assert counts()['custom'] == {'A':2,'B':3}
    pending_columns.discard(table,['custom'])
    assert counts()['custom'] == {'A':1,'B':2}
    frame.loc[0,'custom'] = None
    frame.to_parquet(table,index=False)
    assert counts()['custom'] == {'B':2}


def test_concurrent_requests_share_one_summary_and_corrupt_cache_recovers(tmp_path,monkeypatch):
    monkeypatch.setenv('SPACEMAP_CACHE_DIR',str(tmp_path/'cache'))
    calls=[]
    def build():
        calls.append(1);time.sleep(.05);return {'cells':42}
    def get():
        return summary_cache.cached('parallel',['table'],lambda:[1],build)
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(lambda _:get(),range(4))) == [{'cells':42}]*4
    assert len(calls)==1
    get()['cells']=99
    assert get()['cells']==42
    summary_cache._MEMORY.clear()
    next((tmp_path/'cache/studio-summaries-v1').glob('*.json')).write_text('broken')
    assert get()['cells']==42 and len(calls)==2


def test_job_provenance_cache_tracks_new_and_replaced_results(tmp_path,monkeypatch):
    monkeypatch.setenv('SPACEMAP_CACHE_DIR',str(tmp_path/'cache'))
    job=tmp_path/'outputs/studio/jobs/job1';job.mkdir(parents=True)
    request=job/'request.json';status=job/'status.json'
    request.write_text(json.dumps({'kind':'vortex','settings':{'table':'/cells.parquet','output_col':'custom'}}))
    status.write_text(json.dumps({'status':'done','result':{}}))
    assert overview._job_outputs(tmp_path)[0]['names'][-1]=='custom'
    summary_cache._MEMORY.clear()
    original=Path.read_text
    def read(path,*a,**k):
        assert path not in (request,status), 'Unchanged manifests must not be reparsed after restart'
        return original(path,*a,**k)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'read_text',read)
        assert overview._job_outputs(tmp_path)[0]['kind']=='vortex'
    request.write_text(json.dumps({'kind':'leiden','settings':{'table':'/cells.parquet','output_col':'new'}}))
    assert overview._job_outputs(tmp_path)[0]['kind']=='leiden'
    status.write_text(json.dumps({'status':'running'}))
    assert overview._job_outputs(tmp_path)==[]


def test_metadata_and_flow_scope_do_not_need_summaries_or_cell_rows(setup,monkeypatch):
    from spacemap_gui.studio import phenotyping_inputs, phenotyping_images, flow_gating
    channel,table,root,_=setup
    def forbidden(*a,**k):
        raise AssertionError('Startup metadata must not scan cells or old jobs')
    for name in ('overview','results','method_columns'):
        monkeypatch.setattr(overview,name,forbidden)
    monkeypatch.setattr(flow_gating,'evaluate',forbidden)
    info=phenotyping_inputs.plan(root,{},table,summaries=False)
    assert info['selected_images']==['P11_roi_2'] and info['cells'] is None
    assert phenotyping_images.sources(root,table)[0]['selected_images']==['P11_roi_2']
    found=flow_gating.options(channel,metadata_only=True)
    assert {'M0','M1'} <= set(found['numeric']) and found['n'] is None
    assert found['stats']==[]


def test_model_choices_do_not_deserialize_the_selected_model(tmp_path, monkeypatch):
    from spacemap_gui.studio import reference_setup, reference_models
    folder = tmp_path / 'models'; folder.mkdir()
    identifier = 'a' * 32
    # An artifact need only exist for the menu. Its bytes are intentionally
    # unreadable: opening controls must not load or validate a prediction model.
    (folder / (identifier + '.joblib')).write_bytes(b'not a pickle')
    (folder / (identifier + '.json')).write_text(json.dumps({
        'method': 'Random Forest', 'markers': ['M0'], 'classes': ['Treg'],
        'reusable_examples': 12, 'saved_example_cells': 3}))
    monkeypatch.setattr(reference_models, 'directory', lambda: folder)
    monkeypatch.setattr(reference_setup.model_library, 'location', lambda: folder)
    saved = reference_setup._path(tmp_path); saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({'folder': str(folder), 'model_id': identifier}))
    def forbidden(*a, **k):
        raise AssertionError('Model choices must use metadata only')
    monkeypatch.setattr(reference_models, '_source', forbidden)
    info = reference_setup.options(tmp_path)
    assert info['base_model_id'] == identifier
    assert info['models'][0]['classes'] == ['Treg']
    assert info['models'][0]['reusable_examples'] == 12
