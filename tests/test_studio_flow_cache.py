"""Disposable caches cannot become stale analysis or erase scientific outputs."""
import copy
from pathlib import Path

from spacemap import dataio, labels
from spacemap_gui.studio import flow_cache, flow_gating as flow
from test_studio_commands import setup  # noqa: F401


def test_batched_projection_keeps_saved_text_overlays(setup,monkeypatch):
    channel,table,_,_=setup
    labels.append(table,[{'row':1,'column':'phenotype_ref1','value':'Treg'}])
    reader=flow.data_view._table(str(table))
    original=reader.file
    reads=[]
    class RecordedFile:
        def read(self,**kwargs):
            reads.append(kwargs['columns'])
            return original.read(**kwargs)
    monkeypatch.setattr(reader,'file',RecordedFile())
    batch=reader.project(['M0','M1','batch_id','phenotype_ref1'])
    assert reads==[['M0','M1','batch_id']]
    assert batch['phenotype_ref1'][1].as_py()=='Treg'
    assert batch['M0'].tolist()==[1.,2.,3.,4.,100.]
    reader.project(['M1','M0'])
    assert len(reads)==1


def test_plain_scatter_skips_density_but_density_modes_get_exact_grid(setup):
    channel,_,_,_=setup
    spec={'x':{'column':'M0'},'y':{'column':'M1'},'kind':'scatter','sample':100,'bins':4}
    plain=flow.plot(channel,{'plot':spec})
    assert 'density' not in plain['points'][0]
    density=flow.plot(channel,{'plot':{**spec,'kind':'density_scatter'}})
    assert sum(map(sum,density['points'][0]['density']['z']))==4
    assert flow.plot(channel,{'plot':{**spec,'kind':'contour'}})['cached']


def test_branch_edits_reuse_parent_plot_and_restore_exact_counts_without_cell_reads(setup,monkeypatch):
    channel,_,_,_=setup
    spec={'x':{'column':'M0'},'sample':100,'bins':4,'parent':'root'}
    before=flow.plot(channel,{'plot':spec})
    gate={'id':'new','kind':'range','name':'New','colour':'#123456','parent':'root','x':{'column':'M0'},'bounds':[1,2]}
    flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    flow_cache._MEMORY.clear();flow._CACHE.clear()  # restart
    with monkeypatch.context() as patch:
        def forbidden(*a,**k):raise AssertionError('Unchanged parent plot must reuse its exact cached values')
        patch.setattr(flow,'dataset',forbidden)
        metadata=flow.options(channel,metadata_only=True)
        assert next(s for s in metadata['stats'] if s['id']=='new')['n']==2
        current=flow.plot(channel,{'plot':spec})
        assert current['cached'] and current['histogram']==before['histogram']
        assert current['gates'][0]['id']=='new'
        assert next(s for s in current['stats'] if s['id']=='new')['n']==2
    child=flow.plot(channel,{'plot':{**spec,'parent':'new'}})
    assert not child.get('cached') and child['histogram']['series'][0]['n']==2
    flow.save(channel,{'revision':1,'changes':{'gates':[{**gate,'bounds':[1,3]}]}})
    changed=flow.plot(channel,{'plot':{**spec,'parent':'new'}})
    assert not changed.get('cached') and changed['histogram']['series'][0]['n']==3


def test_new_plot_and_layout_save_need_no_cell_read(setup, monkeypatch):
    channel, _, _, _ = setup
    spec = {'id':'first', 'x':{'column':'M0'}, 'y':{'column':'M1'}, 'sample':100, 'bins':4}
    first = flow.plot(channel, {'revision':0, 'plot':spec})
    flow_cache._MEMORY.clear()  # simulate a fresh process using the disk cache
    def forbidden(*args, **kwargs):
        raise AssertionError('Repeated plots/layout saves must not load or fit cell rows')
    monkeypatch.setattr(flow, 'dataset', forbidden)
    saved = flow.save(channel, {'revision':0, 'changes':{'plots':[spec]}})
    assert saved['workspace']['revision'] == 1
    second = flow.plot(channel, {'revision':1, 'plot':{**spec, 'id':'new', 'style':{'font':20}}})
    assert second['cached'] and second['points'] == first['points']
    assert second['histogram'] == first['histogram']
    assert second['plot']['id'] == 'new' and second['plot']['style']['font'] == 20


def test_cache_keeps_fitted_recipes_and_invalidates_measurements(setup):
    channel, table, _, _ = setup
    spec = {'x':{'column':'M0', 'normalize':'zscore'}, 'sample':100, 'bins':4}
    first = flow.plot(channel, {'revision':0, 'plot':spec})
    flow_cache._MEMORY.clear()
    assert flow.plot(channel, {'plot':first['plot']})['cached']
    gate = {'id':'a','kind':'range','name':'A','colour':'#112233', 'x':{'column':'M0'}, 'bounds':[1,2]}
    flow.save(channel, {'revision':0, 'changes':{'gates':[gate]}})
    fresh = flow.plot(channel, {'revision':1, 'plot':spec})
    assert fresh.get('cached') and any(s['id']=='a' for s in fresh['stats'])
    labels.append(table, [{'row':0, 'column':'M0', 'value':90}])
    fresh = flow.plot(channel, {'revision':1, 'plot':spec})
    assert not fresh.get('cached')
    assert fresh['plot']['x']['fitted'] != first['plot']['x']['fitted']


def test_renaming_reuses_plot_values_with_current_names_and_colours(setup,monkeypatch):
    channel, _, _, _ = setup
    gate = {'id':'a','kind':'range','name':'A','colour':'#112233', 'x':{'column':'M0'}, 'bounds':[1,2]}
    flow.save(channel, {'revision':0,'changes':{'gates':[gate]}})
    spec = {'x':gate['x'],'sample':100,'bins':4,'series':[{'population':'a'}]}
    first = flow.plot(channel, {'plot':spec})
    flow.save(channel, {'revision':1,'changes':{'gates':[{**gate,'name':'Renamed','colour':'#abcdef'}]}})
    monkeypatch.setattr(flow,'fit_plot',lambda *a,**k: (_ for _ in ()).throw(AssertionError('No numerical recomputation for naming')))
    current = flow.plot(channel, {'plot':spec})
    assert current['cached']
    assert current['histogram']['series'][0]['counts'] == first['histogram']['series'][0]['counts']
    assert current['histogram']['series'][0]['name'] == current['gates'][0]['name'] == 'Renamed'
    assert current['histogram']['series'][0]['colour'] == current['gates'][0]['colour'] == '#abcdef'


def test_population_colours_save_without_reading_cells_and_sync_cached_plots(setup,monkeypatch):
    channel, _, _, _ = setup
    gate = {'id':'a','kind':'range','name':'A','colour':'#112233', 'x':{'column':'M0'}, 'bounds':[1,2]}
    flow.save(channel, {'revision':0,'changes':{'gates':[gate]}})
    spec = {'x':gate['x'],'sample':100,'bins':4,'series':[{'population':'root'},{'population':'a'}]}
    first = flow.plot(channel, {'plot':spec})
    membership = flow.load(channel)['membership_file']
    with monkeypatch.context() as patch:
        def forbidden(*a,**k):
            raise AssertionError('Colour edits must not read cell measurements')
        patch.setattr(flow,'dataset',forbidden)
        flow.save(channel, {'revision':1,'changes':{'root_colour':'#6789ab','gates':[{**gate,'colour':'#abcdef'}]}})
        current = flow.plot(channel, {'plot':spec})
    assert current['cached'] and current['points']==first['points']
    assert [s['colour'] for s in current['histogram']['series']]==['#6789ab','#abcdef']
    assert flow.load(channel)['membership_file']==membership
    for identifier, colour in [('root','#6789ab'),('a','#abcdef')]:
        assert flow.select(channel,{'revision':2,'populations':[identifier]})['colour']==colour
    flow.save(channel,{'action':'undo','revision':2})
    assert flow.load(channel)['root_colour']=='#a8b4c4'
    assert flow.load(channel)['gates'][0]['colour']=='#112233'


def test_cache_clear_and_limits_preserve_gates_labels_history_and_exports(setup, monkeypatch):
    channel, table, root, _ = setup
    before_table = table.read_bytes()
    gate = {'id':'a','kind':'range','name':'A','colour':'#112233', 'x':{'column':'M0'}, 'bounds':[1,2]}
    for i in range(5):
        flow.save(channel, {'revision':i, 'changes':{'gates':[{**gate, 'bounds':[1,2+i]}]}})
    _, _, folder = flow.context(channel)
    assert len(list(folder.glob('membership-*.parquet'))) == 2
    monkeypatch.setattr(flow_cache, 'MAX_PLOTS', 2)
    for bins in (4,5,6):
        flow.plot(channel, {'plot':{'x':{'column':'M0'}, 'sample':100, 'bins':bins}})
    assert len(list((folder/'plot-cache').glob('*.json.gz'))) == 2
    labels.append(table, [{'row':0, 'column':'phenotype_ref1', 'value':'Treg'}])
    journal = labels.sidecar_path(table).read_bytes()
    figure = folder/'figures/figure.svg'; figure.parent.mkdir(); figure.write_text('saved figure')
    outside = root/'keep.parquet'; outside.write_text('not a cache')
    link = folder/('membership-'+'f'*24+'.parquet'); link.symlink_to(outside)
    before = copy.deepcopy(flow.load(channel))
    assert flow_cache.state(root)['bytes'] > 0
    assert flow_cache.clear(root)['files'] == 0
    assert flow.load(channel)['gates'] == before['gates']
    assert flow.load(channel)['history'] == before['history']
    assert figure.read_text() == 'saved figure' and outside.read_text() == 'not a cache'
    assert table.read_bytes() == before_table and labels.sidecar_path(table).read_bytes() == journal
    rebuilt = flow.save(channel, {'revision':5, 'changes':{'gates':before['gates']}})
    assert Path(rebuilt['workspace']['membership_file']).is_file()
    assert dataio.load(table, columns=['phenotype_ref1']).iloc[0,0] == 'Treg'


def test_corrupt_or_unwritable_plot_cache_recomputes(setup, monkeypatch):
    channel, _, _, _ = setup
    spec = {'x':{'column':'M0'}, 'sample':100, 'bins':4}
    first = flow.plot(channel, {'plot':spec})
    _, _, folder = flow.context(channel)
    flow_cache._MEMORY.clear()
    next((folder/'plot-cache').glob('*.json.gz')).write_bytes(b'corrupt')
    second = flow.plot(channel, {'plot':spec})
    assert not second.get('cached') and first['histogram'] == second['histogram']
    monkeypatch.setattr(flow_cache, 'MAX_BYTES', 1)
    flow.plot(channel, {'plot':{**spec, 'bins':5}})
    assert not list((folder/'plot-cache').glob('*.json.gz'))
