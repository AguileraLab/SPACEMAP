"""Separate CSV/Parquet spatial inputs, coordinate isolation and guarded publication."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, pending_columns, table_collection as collection
from spacemap_gui.studio import spatial_sources, spatial, tasks, tls, neighbourhoods
from test_studio_commands import setup  # noqa: F401


@pytest.fixture
def sources(setup):  # noqa: F811
    channel, _, root, _ = setup
    from spacemap_gui.studio import phenotyping_inputs as inputs
    (root / 'outputs/rois/second.ome.tiff').touch()
    frame = pd.DataFrame({'label': np.arange(1, 31), 'x_centroid': np.arange(30, dtype=float),
        'y_centroid': np.zeros(30), 'tissue_roi': 1, 'M0': np.arange(30.),
        'phenotype_test': ['B', 'T'] * 15})
    a, b = root/'P11_roi_2_cell_features.csv', root/'second_cell_features.parquet'
    frame.to_csv(a, index=False)
    frame.to_parquet(b, index=False)
    inputs.use_table(channel, a)
    table = spatial_sources.resolve(root, str(a))
    return root, a, b, table


def settings(table, images=None):
    chosen = neighbourhoods.values({})
    chosen.update(phenotype_col='phenotype_test', roi_col='', batch_col='', k_neighbors=5, min_cells=5, n_neighborhoods=2)
    return dict(table=table, markers=['phenotype_test', 'x_centroid', 'y_centroid'],
        output_col='neighborhood_cn', names_col='neighborhood', result='neighbourhoods',
        cell_id_col='cell_id', images=images or ['P11_roi_2', 'second'], image_col='batch_id',
        values=chosen, overwrite_columns=[])


def test_discovery_projected_ids_and_no_merged_measurements(sources):
    root, a, b, table = sources
    info = spatial_sources.overview(root, table, summaries=True)
    assert [(r['roi'], r['cells']) for r in info['rois']] == [('P11_roi_2', 30), ('second', 30)]
    assert info['cells'] == 60
    frame = dataio.load(table, columns=['cell_id', 'batch_id', 'label'])
    assert frame.cell_id.nunique() == 60 and frame.label.nunique() == 30
    assert frame.batch_id.value_counts().to_dict() == {'P11_roi_2': 30, 'second': 30}
    assert set(dataio.columns(table)) >= {'batch_id', 'cell_id', 'phenotype_test'}
    assert spatial.composition(table, 'phenotype_test')['counts'] == [[15,15],[15,15]]
    assert spatial.read(table, ['label'], ['second']).index.tolist() == list(range(30,60))
    assert spatial_sources.resolve(root, b) == table
    assert all(p.name.endswith(collection.SUFFIX) for p in Path(table).parent.iterdir())


def test_inputs_share_exact_images_and_pairs_but_keep_separate_output(sources, setup):
    from spacemap_gui.studio import phenotyping_inputs as inputs
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import api
    root,a,b,_=sources
    channel=setup[0]
    client=TestClient(api.app)
    url=f'/studio/{channel}/spatial/inputs'
    found=client.get(url).json()
    assert {r['image_path'] for r in found['rows']}=={str(root/'outputs/rois/P11_roi_2.ome.tiff'),str(root/'outputs/rois/second.ome.tiff')}
    pheno_output=inputs.folders(root,{},a)['output']
    destination=root/'spatial_custom'
    changed=client.post(url,json={'folders':{'output':str(destination)}})
    assert changed.status_code==200,changed.text
    assert changed.json()['folders']['output']==str(destination)
    assert inputs.folders(root,{},a)['output']==pheno_output
    unmatched=client.post(url,json={'pair':{'image':'second','table':''}}).json()
    assert next(r for r in unmatched['rows'] if r['roi']=='second')['table_path']==''
    assert inputs.image_tables(root,a)['second']==''
    restored=client.post(url,json={'pair':{'image':'second','table':str(b)}}).json()
    assert next(r for r in restored['rows'] if r['roi']=='second')['table_path']==str(b)
    assert spatial._out(root,'density').parent==destination/'density'


def test_filename_matching_preserves_ids_and_refuses_ambiguous_ties():
    from spacemap_gui.studio.phenotyping_inputs import match_table_image
    images=['HCC_SARC_1','HCC_SARC_10','patient_2_region_1']
    assert match_table_image('hcc-sarc-1_cell_features_tagged.csv',images)=='HCC_SARC_1'
    assert match_table_image('patinet_2_region_1_cell_features.csv',images)=='patient_2_region_1'
    assert match_table_image('HCC_SARC_11_cell_features.csv',images)==''
    assert match_table_image('hcc-sarc-1.csv',['HCC_SARC_1','hcc_sarc_1'])==''


def test_method_exports_preserve_original_job_files(tmp_path):
    from spacemap_gui.studio import spatial_exports
    job=tmp_path/'job';job.mkdir()
    (job/'tls_candidates.csv').write_text('image,cells\na,50\n')
    result={'folder':str(job),'counts':{'a':1}}
    output=spatial_exports.publish(tmp_path,'tls_candidates',{'spatial_output':str(tmp_path/'chosen')},job,result)
    dest=Path(output['export_folder'])
    assert dest.parent==tmp_path/'chosen/tls_candidates'
    assert (dest/'tls_candidates.csv').read_bytes()==(job/'tls_candidates.csv').read_bytes()
    assert json.loads((dest/'result.json').read_text())==result


def test_missing_column_report_names_selected_image(sources):
    _, a, b, table = sources
    frame = pd.read_parquet(b).drop(columns='phenotype_test')
    frame.to_parquet(b, index=False)
    with pytest.raises(ValueError, match='second: no phenotype_test'):
        tasks.validate_table(settings(table))
    tasks.validate_table(settings(table, ['P11_roi_2']))
    with pytest.raises(ValueError, match='second: no phenotype_test'):
        spatial.composition(table, 'phenotype_test')


def test_ambiguous_pair_stays_visible_and_is_not_guessed(sources):
    root,a,b,_=sources
    pd.read_csv(a).to_parquet(root/'P11_roi_2_cell_features.parquet',index=False)
    table=spatial_sources.resolve(root,b)
    rows=spatial_sources.overview(root,table)['rois']
    assert rows[0]['roi']=='P11_roi_2' and rows[0]['ready'] is False
    assert rows[1]['roi']=='second' and rows[1]['ready'] is True
    with pytest.raises(ValueError,match='No paired cell table'):
        collection.validate(table,['phenotype_test'],['P11_roi_2'])
    pd.read_parquet(b).to_csv(root/'second_cell_features.csv',index=False)
    empty=spatial_sources.resolve(root,b)
    assert all(not row['ready'] for row in spatial_sources.overview(root,empty)['rois'])


def test_selected_publication_leaves_other_source_and_values_untouched(sources):
    _, a, b, table = sources
    before = (a.read_bytes(), b.read_bytes())
    cfg = settings(table, ['P11_roi_2'])
    tasks.validate_table(cfg)
    cfg['revision'] = tasks.source_revision(cfg)
    ids = dataio.load(table, columns=['cell_id']).cell_id
    done = tasks.save_result(cfg, {'neighborhood_cn': pd.Series(['CN 0']*30+['CN 1']*30)}, ids, lambda *a:None)
    assert done['tables'] == [str(a)]
    assert dataio.load(a, columns=['neighborhood_cn']).neighborhood_cn.tolist() == ['CN 0']*30
    assert 'neighborhood_cn' not in dataio.columns(b)
    assert before == (a.read_bytes(), b.read_bytes())
    # The same name may be applied to an unprocessed image without overwriting A.
    cfg = settings(table, ['second'])
    tasks.validate_table(cfg)
    cfg['revision'] = tasks.source_revision(cfg)
    tasks.save_result(cfg, {'neighborhood_cn': pd.Series(['wrong']*30+['CN 1']*30)}, ids, lambda *a:None)
    assert dataio.load(a, columns=['neighborhood_cn']).neighborhood_cn.eq('CN 0').all()
    assert dataio.load(b, columns=['neighborhood_cn']).neighborhood_cn.eq('CN 1').all()


def test_stale_inputs_refuse_write_and_partial_failure_rolls_back(sources, monkeypatch):
    _, a, b, table = sources
    cfg = settings(table)
    cfg['revision'] = tasks.source_revision(cfg)
    frame = pd.read_parquet(b); frame.loc[0,'M0'] = 999; frame.to_parquet(b,index=False)
    ids = dataio.load(table, columns=['cell_id']).cell_id
    out = {'neighborhood_cn': pd.Series(['CN 0']*60)}
    with pytest.raises(ValueError, match='changed'):
        tasks.save_result(cfg,out,ids,lambda *a:None)
    cfg['revision'] = tasks.source_revision(cfg)
    save = pending_columns.save
    def fail(path,*args,**kwargs):
        if str(path) == str(b):
            raise OSError('simulated full disk')
        return save(path,*args,**kwargs)
    monkeypatch.setattr(pending_columns,'save',fail)
    with pytest.raises(OSError,match='full disk'):
        tasks.save_result(cfg,out,ids,lambda *a:None)
    assert 'neighborhood_cn' not in dataio.columns(a) and 'neighborhood_cn' not in dataio.columns(b)


def test_duplicate_cell_ids_within_one_file_rejected(sources):
    _, _, b, table = sources
    frame = pd.read_parquet(b); frame.loc[0,'label'] = 2; frame.to_parquet(b,index=False)
    with pytest.raises(ValueError,match='second: Cell IDs'):
        tasks.validate_table(settings(table))


def test_neighbourhoods_no_cross_image_neighbours_even_without_grouping(sources, monkeypatch):
    root,a,b,table=sources
    # Identical positions; mixing images would make false B/T neighbourhoods.
    af=pd.read_csv(a);af['phenotype_test']='B';af.to_csv(a,index=False)
    bf=pd.read_parquet(b);bf['phenotype_test']='T';bf.to_parquet(b,index=False)
    cfg=settings(table);cfg['revision']=tasks.source_revision(cfg)
    done=neighbourhoods.run(cfg,root/'nb-job',lambda *a:None)
    raw=np.asarray(done['profile']['raw'])
    assert raw.shape==(2,2) and np.allclose(sorted(raw[:,0]),[0,1])
    assert all('neighborhood_cn' in dataio.columns(p) for p in [a,b])
    assert {r['roi'] for rows in done['where'].values() for r in rows}=={'P11_roi_2','second'}


def test_tls_detects_and_saves_independently_without_combined_table(sources):
    root,a,b,table=sources
    cfg=tls.prepare(root,table,{**tls.DEFAULTS,'phenotype_col':'phenotype_test','t_labels':['T'],
        'min_b':1,'min_t':1,'min_cells':2,'max_gap':4.,'min_fraction':0.,'pixel_um':1.,'roi_col':''},
        ['P11_roi_2'],False)
    folder=root/'tls-job';folder.mkdir()
    done=tls.run(cfg,folder,lambda *a:None)
    assert done['counts']['P11_roi_2']>=1 and done['tables']==[str(a)]
    assert 'TLS' in dataio.columns(a) and 'TLS' not in dataio.columns(b)
    assert pd.read_csv(a).columns.tolist()==pd.read_parquet(b).columns.tolist()


def test_screening_and_click_use_local_rows_in_the_correct_file(sources, monkeypatch):
    from spacemap_gui.studio import screening, flow_gating
    root,a,b,table=sources
    cfg=screening.defaults({'sample_col':'batch_id','image_col':'batch_id','phenotype_col':'phenotype_test',
        'phenotypes':['B','T'],'zone_source':'whole','pixel_um':1.,'min_cells':1,
        'families':['proportion'],'endpoint':{'type':'describe'}})
    prepared=screening.prepare(root,table,cfg,['P11_roi_2','second'])
    done=screening.job(prepared,None,lambda *a:None)
    seen={}
    def capture(channel,path,stamp,selected,*args):
        seen.update(table=path, selected=selected)
        return 'selection-test'
    monkeypatch.setattr(flow_gating,'register_selection',capture)
    hit=done['hits'][0]
    linked=screening.context_cells(root,table,done['run'],'second',hit['feature'],hit['zone'],'test')
    assert linked['table']==str(b) and seen['table']==str(b)
    assert seen['selected']._flow_row.max()<30 and seen['selected']._flow_roi.eq('second').all()
    assert seen['selected']._flow_table.eq(str(b)).all()
    assert (Path(done['folder'])/'figures/proportion.png').is_file()


def test_interaction_randomization_keeps_image_spaces_separate(sources):
    from spacemap_gui.studio import interactions
    root,a,b,table=sources
    af=pd.read_csv(a);af['phenotype_test']='B';af.to_csv(a,index=False)
    bf=pd.read_parquet(b);bf['phenotype_test']='T';bf.to_parquet(b,index=False)
    chosen=interactions.values({});chosen.update(phenotype_col='phenotype_test',batch_col='',roi_col='',
        radius_um=3.,pixel_um=1.,min_cells=5,n_permutations=50)
    done=interactions.run({'table':table,'values':chosen,'images':['P11_roi_2','second'],'image_col':'batch_id'},root/'cci',lambda *a:None)
    cross=[p for p in done['pairs'] if {p['a'],p['b']}=={'B','T'}]
    assert not cross or all(p['observed']==0 for p in cross)
