"""Provenance links stay reusable and do not promote predictions to ground truth."""
import json

from spacemap_gui.studio import annotation, annotation_edits as edits, annotation_workflow as workflow
from spacemap_gui.studio import artifact_learning as learning
from test_artifact_learning import polygon, slide  # noqa: F401


def test_index_settings_histories_models_and_source_files(tmp_path, slide):  # noqa: F811
    image, _ = slide
    output = tmp_path / 'annotations'
    annotation.save_step(tmp_path, {'folders': {'images': str(tmp_path), 'output': str(output)}, 'label': 'My tissue'})
    target = annotation.valid_tissue(output, 'first')
    feature = polygon()
    feature['properties'].update(tissue_source='source.geojson', tissue_source_class='tissue', tissue_source_revision='abc')
    edits.replace(target, [feature], edits.revision(target), 'Imported Tissue', kind='tissue')
    learning.snapshot(tmp_path, 'clot', image, 2, [polygon(0, 0, 64, 128)], [], [polygon()], True)
    version = learning.train(learning.training_request(tmp_path, 'clot'), lambda *a: None)['version']
    before = target.read_bytes()
    data = workflow.record(tmp_path, image=image, target=target, event={'operation': 'test', 'parameters': {'gap': 3}})
    row = data['images']['first.ome.tiff']
    assert row['final_review'] == 'not_declared'
    asset = data['assets'][row['layers']['tissue_original']]
    assert asset['history'] and asset['sources'][0]['tissue_source_class'] == 'tissue'
    assert asset['sha256'] == edits.revision(target) and target.read_bytes() == before
    assert 'sha256' not in data['assets']['first.ome.tiff']
    model = data['artifact_libraries']['clot']['models'][0]
    assert model['id'] == version['id'] and (tmp_path / model['file']).is_file()
    snap = data['artifact_libraries']['clot']['snapshots'][version['examples'][0]]
    assert (tmp_path / snap['annotations']).is_file() and (tmp_path / snap['samples']).is_file()
    event = json.loads((tmp_path / data['latest_event']).read_text())
    assert event['parameters']['gap'] == 3 and event['saved_revision'] == asset['sha256']
    old_snapshot = data['settings_snapshot']
    annotation.save_step(tmp_path, {'label': 'Changed label'})
    updated = workflow.record(tmp_path, image=image, event={'operation': 'settings'})
    assert updated['settings_snapshot'] != old_snapshot and (tmp_path / old_snapshot).is_file()
    assert updated['artifact_libraries']['clot']['snapshots'][version['examples'][0]]['annotations'] == snap['annotations']
    assert 'annotations' not in json.loads((tmp_path / updated['latest_event']).read_text())
    # No arbitrary root config fields are copied into the shareable record.
    from spacemap_gui.studio.settings import config_path
    config = json.loads(config_path(tmp_path).read_text()); config['api_key'] = 'secret-do-not-export'
    config_path(tmp_path).write_text(json.dumps(config))
    data = workflow.record(tmp_path)
    assert 'secret-do-not-export' not in json.dumps(data)
    assert data['future_workflow']['status'] == 'roadmap_not_implemented'


def test_quality_branch_records_original_blur_snapshot_location(tmp_path, slide):  # noqa: F811
    image, _ = slide
    annotation.save_step(tmp_path, {'folders': {'images': str(tmp_path), 'output': str(tmp_path / 'annotations')}})
    learning.snapshot(tmp_path, 'blur', image, 2, [polygon(0, 0, 64, 128)], [], [polygon()], True)
    parent = learning.train(learning.training_request(tmp_path, 'blur'), lambda *a: None)['version']
    child = learning.train(learning.training_request(tmp_path, 'debris', base_version='blur:' + parent['id']), lambda *a: None)['version']
    data = workflow.record(tmp_path, image=image)
    library = data['artifact_libraries']['debris']
    assert library['models'][0]['parent_version'] == 'blur:' + parent['id']
    source = library['snapshots'][child['examples'][0]]
    assert source['source_part'] == 'blur'
    assert (tmp_path / source['samples']).is_file()
    assert '/blur/' in source['samples'] and '/blur/' in source['annotations']
