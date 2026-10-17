"""Keep exact sampling, borders and labels while avoiding marker-cache thrashing."""
import numpy as np
import pytest

from spacemap_gui.studio import artifact_learning as learning, artifact_quality as quality, artifact_sampling as sampling
from test_studio_artifacts import marker_image


def image(path, size=530):
    rng=np.random.default_rng(9)
    data=rng.integers(1,2000,(5,size,size),dtype=np.uint16)
    data[:3]%=255
    marker_image(path,data,['HE_Red','HE_Green','HE_Blue','CD3','DAPI'])
    return path


@pytest.mark.parametrize('schema',['legacy','quality_v1'])
def test_channel_order_and_exact_sampled_features(tmp_path,schema,monkeypatch):
    reader=learning._extractor(image(tmp_path/'test.ome.tiff'),1,schema)
    rng=np.random.default_rng(8)
    points=np.vstack([rng.integers(0,530,(220,2)),[[0,0],[529,529],[0,529],[529,0],[256,256]]])
    expected=np.empty((len(points),len(reader.names)),np.float32)
    size=getattr(reader,'tile_size',learning.TILE)
    for tx,ty in np.unique(points//size,axis=0):
        ids=np.flatnonzero((points//size==(tx,ty)).all(axis=1));x,y=int(tx*size),int(ty*size)
        expected[ids]=reader.tile(x,y,min(size,530-x),min(size,530-y))[points[ids,1]-y,points[ids,0]-x]
    calls=[];original=reader.read
    def read(channel,*args,**kwargs):
        calls.append(channel);return original(channel,*args,**kwargs)
    monkeypatch.setattr(reader,'read',read)
    actual,execution=sampling.sample(reader,points,device='cpu')
    np.testing.assert_allclose(actual,expected,atol=2e-6)
    marker_calls=[i for i in calls if i in reader.markers]
    assert marker_calls==sorted(marker_calls)
    assert execution['windows']>1 and execution['order']=='channel-major'


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_torch_matches_scipy_borders_tiny_arrays_and_blank_regions(device):
    torch=pytest.importorskip('torch')
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA device not available')
    rng=np.random.default_rng(99)
    kernel=sampling.TorchMeasurements(device)
    for shape in ((1,2),(4,5),(65,73)):
        arrays=[rng.random(shape,dtype=np.float32),np.zeros(shape,np.float32),np.full(shape,2.3,np.float32)]
        arrays[0][shape[0]//2:,shape[1]//2:]=0
        yy,xx=np.indices(shape)
        pos=(np.repeat(np.arange(3),yy.size),np.tile(yy.ravel(),3),np.tile(xx.ravel(),3))
        for fine,coarse in ((3,7),(7,19)):
            with np.errstate(invalid='raise'):
                expected=np.concatenate([np.stack(sampling.measurements(a,fine,coarse),axis=-1).reshape(-1,9) for a in arrays])
            found=kernel.selected(arrays,pos,fine,coarse)
            assert np.isfinite(found).all()
            np.testing.assert_allclose(found,expected,atol=2e-5,rtol=2e-5)


def test_gpu_failure_keeps_all_examples_on_cpu(tmp_path,monkeypatch):
    reader=quality.extractor(image(tmp_path/'test.ome.tiff',80),1)
    points=np.array([[20,20],[79,79],[40,30]])
    expected,_=sampling.sample(reader,points,device='cpu')
    class Unavailable:
        def selected(self,*args):raise RuntimeError('out of memory')
    monkeypatch.setattr(sampling,'backend',lambda *a:(Unavailable(),{'device':'cuda'}))
    actual,execution=sampling.sample(reader,points)
    np.testing.assert_array_equal(actual,expected)
    assert execution['device']=='cpu' and execution['fallback']=='out of memory'


def test_prediction_uses_direct_reads_only_for_panels_exceeding_cache(tmp_path,monkeypatch):
    from spacemap.viewer import sources
    reader=quality.extractor(image(tmp_path/'test.ome.tiff',80),1)
    expected=reader.tile(0,0,80,80)
    reader.read_order='tile-major'
    monkeypatch.setattr(sources,'_WHOLE_CACHE_BYTES',1)
    monkeypatch.setattr(reader.source,'_whole_level',lambda *a:pytest.fail('Prediction thrashed the whole-channel cache'))
    np.testing.assert_array_equal(reader.tile(0,0,80,80),expected)


def test_cancel_during_preparation_does_not_publish_model_or_snapshot(tmp_path):
    from test_artifact_learning import polygon
    class Cancelled(Exception):pass
    def report(message,progress):
        if message.startswith('Reading marker'):raise Cancelled()
    with pytest.raises(Cancelled):
        learning.snapshot(tmp_path,'debris',image(tmp_path/'test.ome.tiff',80),1,
            [polygon(1,1,35,75)],[polygon(40,1,75,75)],[polygon(0,0,80,80)],False,report=report,device='cpu')
    assert not learning.folder(tmp_path,'debris').exists()


def test_worker_freezes_historical_examples_before_preparation(tmp_path,monkeypatch):
    from spacemap_gui.studio import tasks
    from test_artifact_model_versions import submission
    first=submission(tmp_path,'debris','first.tiff',1)
    old=learning.train(learning.training_request(tmp_path,'debris'),lambda *a:None)['version']
    request=learning.training_request(tmp_path,'debris',base_version=old['id'])
    request['capture']={'image':'new.tiff'}
    def capture(*args,report,**kwargs):
        # A different image enters the current library after the request was made.
        submission(tmp_path,'debris','unrelated.tiff',2)
        current=submission(tmp_path,'debris','new.tiff',3)
        report('Reading training examples',.5)
        return {'snapshot_id':current['id']}
    monkeypatch.setattr(learning,'snapshot',capture)
    progress=[]
    result=tasks.train_artifact_model(request,tmp_path,lambda message,value:progress.append(value))
    assert result['version']['examples']==[first['id'],f'{3:032x}']
    assert result['version']['parent_version']==old['id']
    assert progress==sorted(progress) and progress[0]<.8 and progress[-1]>.8


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_existing_forest_gpu_evaluation_keeps_splits_probabilities_and_ties(device):
    torch=pytest.importorskip('torch')
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA device not available')
    from sklearn.ensemble import RandomForestClassifier
    from spacemap_gui.studio.artifact_forest import ForestPredictor
    rng=np.random.default_rng(58)
    x=rng.random((600,5),dtype=np.float32);y=(x[:,0]+x[:,3]>.8).astype(np.uint8)
    forest=RandomForestClassifier(n_estimators=12,max_depth=7,min_samples_leaf=2,random_state=42,n_jobs=2).fit(x,y)
    # Sample the split boundary and its immediate float32 neighbours as well.
    tests=[x]
    for tree in forest.estimators_:
        chosen=np.flatnonzero(tree.tree_.feature>=0)
        batch=np.tile(x[0],(len(chosen)*3,1))
        for i,n in enumerate(chosen):
            f=tree.tree_.feature[n];v=np.float32(tree.tree_.threshold[n])
            batch[3*i:3*i+3,f]=[np.nextafter(v,np.float32(-np.inf)),v,np.nextafter(v,np.float32(np.inf))]
        tests.append(batch)
    samples=np.vstack(tests)
    predictor=ForestPredictor(forest,device)
    for cut in (.05,.5,.95):
        np.testing.assert_array_equal(predictor.predict(samples,cut),forest.predict_proba(samples)[:,1]>=cut)
