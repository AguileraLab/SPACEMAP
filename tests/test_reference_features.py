import pandas as pd
import pytest

from spacemap_gui.studio import reference_features, manual_reference, auto_reference
from spacemap_gui.viewer import registry
from test_studio_manual_reference import study  # noqa: F401
from test_studio_commands import setup  # noqa: F401


def test_explicit_features_are_independent_of_visible_image_and_source_labels(study):
    channel, table, spec = study
    frame = pd.read_parquet(table)
    frame['area'] = [10,20,30,40,50]
    frame['eccentricity'] = [.1,.2,.3,.4,.5]
    frame['HE_Red'] = [1.,2.,3.,4.,5.]
    frame['patient'] = [1,1,2,2,3]
    frame.to_parquet(table,index=False)
    found = reference_features.options(channel)
    groups = {item['name']:item['group'] for item in found['features']}
    assert groups['area'] == groups['eccentricity'] == 'Morphology'
    assert groups['HE_Red'] == 'H&E' and groups['patient'] == 'Other numeric'
    assert not set(groups) & {'cell_id','label','batch_id','prediction','x_centroid'}
    assert 'patient' not in found['selected']
    selected = ['M0','M1','area','eccentricity','HE_Red']
    reference_features.save(channel, selected)
    # An image with only M0, and a clustering panel with just M0, must not
    # change the trained feature list or its units.
    spec['layers'][0]['channels'] = [{'name':'M0','index':0,'visible':True}]
    registry.publish(channel,spec,project=registry.session_project(channel))
    options = manual_reference.training_settings(channel,'prediction')
    assert options['markers'] == selected and options['allow_missing']
    assert options['feature_kinds']['area'] == 'Morphology'
    assert reference_features.options(channel)['selected'] == selected
    for bad in ([],['cell_id'],['prediction'],['not here']):
        with pytest.raises(ValueError,match='Choose at least'):
            reference_features.save(channel,bad)


def test_api_feature_save_updates_enabled_auto_training_and_read_only_guard(study):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    channel, _, _ = study
    auto_reference.configure(channel,'prediction',True)
    route='/studio/'+channel+'/reference_features'
    with TestClient(app) as client:
        response=client.post(route,json={'features':['M1'],'allow_missing':False})
        assert response.status_code == 200, response.text
        assert response.json()['auto_reference']['enabled']
        project,key=auto_reference._context(channel)
        settings=auto_reference._read(project['root'])[key]['settings']
        assert settings['markers']==['M1'] and not settings['allow_missing']
        registry.publish(channel,study[2],project={**registry.session_project(channel),'read_only':True})
        assert client.get(route).status_code==200
        assert client.post(route,json={'features':['M0']}).status_code==400
