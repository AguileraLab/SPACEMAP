"""Shared Classic settings, reference F1 weights and cohort application of saved models."""
import json
import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, pending_columns, ml_mr_utils
from spacemap_gui.studio import mr_ml, reference_models, tasks
from spacemap_gui.studio.settings import revision
from test_studio_reference_models import training  # noqa: F401
from test_mr_ml_reference_only import measurements  # noqa: F401


def test_classic_settings_round_trip_keeps_classic_only_fields(tmp_path):
    path=mr_ml.cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'n_estimators':731,'ground_truth_col':'phenotype_ref2','input_path':'keep me'}))
    before=mr_ml.values(tmp_path)
    assert before['n_estimators']==731
    result=mr_ml.save(tmp_path, {'n_estimators':91,'workflow':'saved_model'}, revision(before))
    assert json.loads(path.read_text())['input_path']=='keep me'
    assert result['n_estimators']==91
    with pytest.raises(ValueError,match='another window'):
        mr_ml.save(tmp_path, {'n_estimators':20}, revision(before))


def test_classic_parameters_and_per_phenotype_f1_used(measurements, tmp_path, monkeypatch):
    frame,path,_=measurements
    truth=np.where(frame.area<100,'A','B')
    frame['phenotype_good']=truth
    frame['phenotype_wrong']=np.where(truth=='A','B','A')
    frame.to_parquet(path,index=False)
    chosen={**mr_ml.DEFAULTS,'markers':['M0','M1'],'morphology_cols':['area','eccentricity'],
            'feature_weights':{'markers':1.,'morphology':1.,'compartments':0.},
            'phenotyping_methods':['phenotype_good','phenotype_wrong'],'n_estimators':23,
            'random_state':19,'test_size':.25,'enable_filter':False}
    calls=[]
    create=ml_mr_utils._create_classifier
    def classifier(method,trees,seed):
        calls.append((method,trees,seed))
        return create(method,trees,seed)
    monkeypatch.setattr(ml_mr_utils,'_create_classifier',classifier)
    settings=mr_ml.job_settings(tmp_path,path,{'cell_id':'cell_id','roi':'batch_id'},chosen)
    result=tasks.mr_ml(settings,tmp_path/'weighted',lambda *args:None)
    assert calls and all(call==('random_forest',23,19) for call in calls)
    assert result['reference_f1']=={'A':{'phenotype_good':1.,'phenotype_wrong':0.},'B':{'phenotype_good':1.,'phenotype_wrong':0.}}
    got=dataio.load(path)
    assert (got[chosen['output_phenotype_col']]==truth).mean()>.95
    assert (tmp_path/'weighted/results/training_reference_f1_weights.csv').is_file()


def model_job(root, table, model_id, **changes):
    chosen={**mr_ml.DEFAULTS,'workflow':'saved_model','model_id':model_id,'enable_filter':False,**changes}
    return mr_ml.job_settings(root,table,{'cell_id':'cell_id','roi':'batch_id'},chosen)


def test_saved_model_predicts_both_rois_without_fitting(training, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    table,settings,root=training
    trained=reference_models.train(settings,root/'train',lambda *args:None)
    model_id=trained['models'][0]['model_id']
    for cls in (RandomForestClassifier,ExtraTreesClassifier):
        monkeypatch.setattr(cls,'fit',lambda *args,**kwargs:pytest.fail('Applying a complete-panel saved model must not refit'))
    before=table.read_bytes()
    job=model_job(root,table,model_id,preserve_reference_labels=False)
    result=tasks.mr_ml(job,root/'predict',lambda *args:None)
    assert result['scored_cells']==120 and result['model_id']==model_id
    found=dataio.load(table)
    assert found.groupby('batch_id')[job['confidence_column']].count().to_dict()=={'roi1':60,'roi2':60}
    assert (found.loc[found.batch_id=='roi1',job['output_col']]=='A').all()
    assert (found.loc[found.batch_id=='roi2',job['output_col']]=='B').all()
    assert before==table.read_bytes() and len(pending_columns.pending(table))==3
    with pytest.raises(ValueError,match='already exists'):
        model_job(root,table,model_id)


@pytest.mark.parametrize('missing_column', [True, False])
def test_saved_model_transfer_missing_feature_and_no_reference(training, missing_column):
    table,settings,root=training
    model_id=reference_models.train(settings,root/'train',lambda *args:None)['models'][0]['model_id']
    target=root/'other.parquet'
    frame=dataio.load(table).drop(columns=['phenotype_ref1'])
    if missing_column:
        frame=frame.drop(columns=['M1'])
    else:
        frame.loc[frame.batch_id=='roi2','M1']=np.nan
    frame.to_parquet(target,index=False)
    job=model_job(root,target,model_id)
    result=tasks.mr_ml(job,root/'predict',lambda *args:None)
    assert result['scored_cells']==120 and result['missing_features']==(['M1'] if missing_column else [])
    assert result['retained_reference_cells']==0
    assert dataio.load(target)[job['confidence_column']].between(0,1).all()


def test_saved_model_preserves_manual_labels_and_excludes_filtered_rows(training):
    table,settings,root=training
    model_id=reference_models.train(settings,root/'train',lambda *args:None)['models'][0]['model_id']
    job=model_job(root,table,model_id,enable_filter=True,filter_col='batch_id',filter_val='roi1')
    result=tasks.mr_ml(job,root/'predict',lambda *args:None)
    found=dataio.load(table)
    known=ml_mr_utils._valid_reference_mask(found.phenotype_ref1,['Others']) & (found.batch_id=='roi1')
    assert found.loc[known,job['output_col']].tolist()==found.loc[known,'phenotype_ref1'].tolist()
    assert found.loc[known,job['confidence_column']].isna().all()
    assert found.loc[found.batch_id=='roi2',job['output_col']].isna().all()
    assert result['retained_reference_cells']==int(known.sum())


def test_invalid_training_reference_votes_and_feature_overwrite_rejected(measurements,tmp_path):
    _,path,_=measurements
    chosen={**mr_ml.DEFAULTS,'markers':['M0','M1'],'enable_filter':False,'phenotyping_methods':['phenotype_ref1']}
    with pytest.raises(ValueError,match='independent'):
        mr_ml.job_settings(tmp_path,path,{},chosen)
    with pytest.raises(ValueError):
        mr_ml.check({'n_estimators':float('nan')})


def test_custom_reference_and_external_validation_remain_disjoint(measurements,tmp_path,monkeypatch):
    from spacemap.viewer import reference
    frame,path,_=measurements
    frame['manual_review']=frame['phenotype_ref1']
    frame['held_out']=None
    # Two overlaps must be removed, plus five independent labels of each class.
    frame.loc[[0,25,26,27,28,29],'held_out']='A'
    frame.loc[[50,75,76,77,78,79],'held_out']='B'
    frame.to_parquet(path,index=False)
    reference.register_column(path,'manual_review')
    chosen={**mr_ml.DEFAULTS,'ground_truth_col':'manual_review','validation_ground_truth_col':'held_out',
            'markers':['M0','M1'],'morphology_cols':['area','eccentricity'],
            'feature_weights':{'markers':1.,'morphology':1.,'compartments':0.},
            'n_estimators':20,'enable_filter':False}
    captured=[]
    pipeline=ml_mr_utils.run_ml_mr_analysis
    def inspect(*args,**kwargs):
        source=kwargs['preloaded_df']
        trained=ml_mr_utils._valid_reference_mask(source['manual_review'],['Others'])
        held=ml_mr_utils._valid_reference_mask(source['held_out'],['Others'])
        assert not (trained&held).any() and held.sum()==10
        captured.append(True)
        return pipeline(*args,**kwargs)
    monkeypatch.setattr(ml_mr_utils,'run_ml_mr_analysis',inspect)
    settings=mr_ml.job_settings(tmp_path,path,{'cell_id':'cell_id'},chosen)
    tasks.mr_ml(settings,tmp_path/'validation',lambda *args:None)
    assert captured and pd.read_parquet(path)['held_out'].notna().sum()==12
    with pytest.raises(ValueError,match='independent'):
        mr_ml.job_settings(tmp_path,path,{}, {**chosen,'phenotyping_methods':['manual_review']})
