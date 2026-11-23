"""Cleanup never treats a whole job or a pending-results store as disposable."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from spacemap_gui.studio import storage


def job(root, name, kind='leiden', status='done', result=None, settings=None):
    folder=root/'outputs/studio/jobs'/name
    folder.mkdir(parents=True)
    (folder/'request.json').write_text(json.dumps({'kind':kind,'settings':settings or {}}))
    (folder/'status.json').write_text(json.dumps({'kind':kind,'status':status,'result':result or {}}))
    return folder


def test_cleanup_preserves_saved_labels_heatmaps_and_unique_outputs(tmp_path):
    table=tmp_path/'cells.csv';table.write_text('cell_id,phenotype_leiden_clusters\na,Cluster 0\nb,Cluster 1\n')
    folder=job(tmp_path,'a'*32,result={'table':str(table),'column':'phenotype_leiden_clusters','profile':{'version':1}},
        settings={'pipeline':{'markers_selected':['M0'],'norm_z1':True}})
    pd.DataFrame({'leiden_cluster':[0,1],'M0_norm':[1.,2.]}).to_parquet(folder/'results.parquet')
    unique=folder/'only-result.geojson';unique.write_text('only copy')
    original=table.read_bytes()
    plan=storage.preview(tmp_path)
    assert plan['groups'][0]['count']==1
    result=storage.clean(tmp_path,plan,['leiden'])
    assert result['cleaned']==1 and result['bytes']>0 and not result['skipped']
    assert table.read_bytes()==original and unique.read_text()=='only copy'
    saved=json.loads((folder/'status.json').read_text())['result']['profile']
    assert saved['version']==2 and set(saved['summaries'])=={'median','mean'}
    assert not (folder/'results.parquet').exists()


def test_active_missing_and_changed_results_are_kept(tmp_path):
    table=tmp_path/'cells.csv';table.write_text('id,result\n1,A\n')
    for name,status,column in [('a','running','result'),('b','done','not_saved'),('c','failed','result'),('d','done','result')]:
        folder=job(tmp_path,name*32,status=status,result={'table':str(table),'column':column})
        (folder/'results.parquet').write_bytes(b'only output')
    plan=storage.preview(tmp_path)
    assert len(plan['items'])==1 and plan['items'][0]['path'].endswith('d'*32+'/results.parquet')
    path=Path(plan['items'][0]['path']);path.write_bytes(b'changed since preview')
    result=storage.clean(tmp_path,plan,['leiden'])
    assert result['cleaned']==0 and len(result['skipped'])==1
    assert len(list((tmp_path/'outputs/studio/jobs').glob('*/results.parquet')))==4


def test_history_requires_saved_final_version_and_keeps_current_and_lock(tmp_path):
    base=tmp_path/'outputs/studio/imported_annotations/image'
    history=base/'studio_history/abcd_test.geojson';(history/'states').mkdir(parents=True);(history/'events').mkdir()
    content=b'{"features":[]}'
    digest=hashlib.sha256(content).hexdigest()
    (history/'states'/f'{digest}.json').write_bytes(content)
    (history/'events/event.json').write_text(json.dumps({'after':digest}))
    assert not storage.preview(tmp_path)['items']
    archived=tmp_path/'outputs/studio/deleted/image/20261023_120000_abcd_test.geojson';archived.parent.mkdir(parents=True);archived.write_bytes(content)
    (history/'edit.lock').touch();inode=(history/'edit.lock').stat().st_ino
    plan=storage.preview(tmp_path);assert len(plan['items'])==1
    result=storage.clean(tmp_path,plan,['history'])
    assert result['cleaned']==1 and archived.read_bytes()==content
    assert not list((history/'states').iterdir()) and not list((history/'events').iterdir())
    assert (history/'edit.lock').stat().st_ino==inode


def test_saved_training_request_is_compacted_only_with_saved_model(tmp_path):
    settings={'part':'clot','captures':[{'image':'image.tiff','positives':[{'geometry':'x'*1100000}],'remaining_good':True}], 'examples':[{'id':'example'}]}
    folder=job(tmp_path,'a'*32,'train_artifact_model',result={'version':{'id':'model1'}},settings=settings)
    assert not storage.preview(tmp_path)['items']
    home=tmp_path/'outputs/studio/artifact_models/clot';home.mkdir(parents=True)
    (home/'model1.joblib').write_bytes(b'model weights');(home/'versions.json').write_text('[{"id":"model1"}]')
    result=storage.clean(tmp_path,storage.preview(tmp_path),['training'])
    assert result['cleaned']==1 and result['bytes']>1000000
    saved=json.loads((folder/'request.json').read_text())
    assert saved['settings']['examples']==[{'id':'example'}]
    assert saved['settings']['captures'][0]['positives_saved_count']==1
    assert (home/'model1.joblib').read_bytes()==b'model weights'


def test_performance_inventory_separates_parts_and_never_cleans_them(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root=tmp_path/'project';root.mkdir()
    shared=tmp_path/'shared'
    monkeypatch.setenv('SPACEMAP_CACHE_DIR',str(shared))
    monkeypatch.delenv('NUMBA_CACHE_DIR',raising=False)
    monkeypatch.delenv('CELLPOSE_LOCAL_MODELS_PATH',raising=False)
    flow=root/'outputs/studio/flow_gating'/('a'*20)
    paths={
        flow/'plot-cache'/('b'*64+'.json.gz'):b'plot cache',
        flow/('membership-'+'c'*24+'.parquet'):b'memberships',
        flow/'workspace.json':b'saved gates, never cache',
        root/'outputs/viewer_cache/cell_labels/image.cell_labels.tif':b'cell pyramid',
        root/'outputs/viewer_cache/cell_labels/roi_tables/image.parquet':b'image table',
        root/'outputs/phenotyping/leiden_cache/res_abc.parquet':b'normalized cells',
        root/'outputs/phenotyping/cache/leiden_params.json':b'settings, never cache',
        shared/'studio-summaries-v1/abc.json':b'summary',
        shared/'numba_cache/module/function.nbc':b'compiled code',
        shared/'cellpose_cache/model':b'model resource',
    }
    for path,content in paths.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    source=root/'measurements.csv';source.write_text('id\n1\n')
    twin=root/'measurements.csv.parquet'
    pq.write_table(pa.table({'id':[1]}).replace_schema_metadata({b'spacemap_source':b'fingerprint'}),twin)
    shared_twins=shared/'fast_copies';shared_twins.mkdir()
    pq.write_table(pa.table({'id':[1]}).replace_schema_metadata({b'spacemap_source':b'fingerprint'}),shared_twins/'source.hash.parquet')
    (shared_twins/'source.hash.parquet.labels.parquet').write_bytes(b'saved labels')
    (root/'outputs/viewer_cache/cell_labels/roi_tables/image.parquet.labels.parquet').write_bytes(b'viewer labels')
    # A Parquet suffix, pending columns or labels alone never make a cache.
    (root/'output.csv').write_text('id\n1\n')
    pq.write_table(pa.table({'id':[1]}),root/'output.csv.parquet')
    (root/'measurements.csv.pending').mkdir();(root/'measurements.csv.pending/result.parquet').write_bytes(b'saved result')
    (root/'measurements.csv.labels.parquet').write_bytes(b'saved labels')
    (root/'outputs/viewer_cache/cell_labels/linked.cell_labels.tif').symlink_to(source)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    inventory=storage.caches(root)
    groups={g['id']:g for g in inventory['groups']}
    assert groups['flow_plots']['bytes']==len(b'plot cache')
    assert groups['flow_memberships']['bytes']==len(b'memberships')
    assert groups['cell_labels']['count']==1 and groups['roi_tables']['count']==1
    assert groups['table_copies']['items']==[{'name':twin.name,'path':str(twin),'bytes':twin.stat().st_size}]
    assert groups['shared_tables']['count']==1 and groups['shared_tables']['items'][0]['name']=='source.hash.parquet'
    assert groups['summaries']['scope']=='Shared' and groups['numba']['count']==1 and groups['cellpose']['count']==1
    assert inventory['preserved'] and inventory['bytes']==sum(g['bytes'] for g in groups.values())
    assert all(g['purpose'] and g['impact'] and Path(g['path']).is_absolute() for g in groups.values())
    listed={i['path'] for g in groups.values() for i in g['items']}
    assert str(flow/'workspace.json') not in listed and not any('.pending/' in p for p in listed)
    plan=storage.preview(root);assert not plan['items']
    assert storage.clean(root,plan,list(storage.NAMES))['bytes']==0
    assert all(p.read_bytes()==content for p,content in before.items())
