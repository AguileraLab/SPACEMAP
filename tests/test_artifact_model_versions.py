"""Model branches must use historical labels, never silently the newest library."""
import json
from hashlib import sha256

import numpy as np
import pytest

from spacemap_gui.studio import artifact_learning as learning


def submission(root, part, image, number, flipped=False):
    home = learning.folder(root, part)
    home.mkdir(parents=True, exist_ok=True)
    identifier = f'{number:032x}'
    x = np.linspace(0, 1, 80, dtype=np.float32).reshape(-1, 1)
    y = (x[:, 0] > .5).astype(np.uint8)
    np.savez_compressed(home / (identifier + '.npz'), x=x, y=1-y if flipped else y)
    record = dict(id=identifier, image=image, image_key=str(root/image), names=['marker:cd3'],
                  scale=2, positive=40, negative=40, remaining_good=False, created='2026-09-29T00:00:00+00:00')
    (home / (identifier + '.json')).write_text(json.dumps({**record, 'annotations': {'test': number}}))
    library = learning._read(home / 'examples.json', {})
    library[sha256(record['image_key'].encode()).hexdigest()] = record
    (home / 'examples.json').write_text(json.dumps(library))
    return record


@pytest.mark.parametrize('part', list(learning.NAMES))
def test_older_branch_selection_delete_restore_and_original_weights(tmp_path, part):
    report = lambda *a: None
    first = submission(tmp_path, part, 'first.tiff', 1)
    old = learning.train(learning.training_request(tmp_path, part), report)['version']
    home = learning.folder(tmp_path, part)
    model_file = home / (old['id'] + '.joblib')
    original = model_file.read_bytes()
    changed = submission(tmp_path, part, 'first.tiff', 2, flipped=True)
    later_image = submission(tmp_path, part, 'later.tiff', 3)
    newer = learning.train(learning.training_request(tmp_path, part), report)['version']
    assert set(newer['examples']) == {changed['id'], later_image['id']}
    selected = learning.manage_model(tmp_path, part, 'select_model', old['id'])
    assert selected['selected_version'] == old['id']
    # A fresh state read (page reload / another image) keeps the older selection.
    assert learning.state(tmp_path, part)['selected_version'] == old['id']
    exact = learning.training_request(tmp_path, part, base_version=old['id'])
    assert [e['id'] for e in exact['examples']] == [first['id']]
    extra = submission(tmp_path, part, 'new.tiff', 4)
    branch = learning.training_request(tmp_path, part, base_version=old['id'], current_snapshot=extra['id'])
    assert {e['id'] for e in branch['examples']} == {first['id'], extra['id']}
    replacement = learning.training_request(tmp_path, part, base_version=old['id'], current_snapshot=changed['id'])
    assert [e['id'] for e in replacement['examples']] == [changed['id']]
    child = learning.train(branch, report)['version']
    assert child['parent_version'] == old['id'] and child['current_snapshot'] == extra['id']
    assert child['id'] not in (old['id'], newer['id']) and child['images'] == 2
    assert model_file.read_bytes() == original
    # Cache must not bypass deletion, while already prepared training jobs remain usable.
    learning.model(tmp_path, part, old['id'])
    deleted = learning.manage_model(tmp_path, part, 'delete_model', old['id'])
    assert old['id'] not in {v['id'] for v in deleted['versions']}
    assert deleted['deleted_versions'][0]['id'] == old['id']
    with pytest.raises(ValueError, match='saved model'):
        learning.model(tmp_path, part, old['id'])
    with pytest.raises(ValueError, match='unavailable'):
        learning.training_request(tmp_path, part, base_version=old['id'])
    assert [e['id'] for e in exact['examples']] == [first['id']]
    restored = learning.manage_model(tmp_path, part, 'restore_model', old['id'])
    assert restored['selected_version'] == old['id'] and not restored['deleted_versions']
    assert model_file.read_bytes() == original
    assert learning.training_request(tmp_path, part, base_version=old['id'])['examples'][0]['id'] == first['id']
    assert learning.state(tmp_path, 'blur' if part != 'blur' else 'clot')['versions'] == []


def test_missing_snapshot_does_not_fall_back_to_latest_examples(tmp_path):
    first = submission(tmp_path, 'clot', 'first.tiff', 1)
    model = learning.train(learning.training_request(tmp_path, 'clot'), lambda *a: None)['version']
    home = learning.folder(tmp_path, 'clot')
    (home / (first['id'] + '.npz')).unlink()
    submission(tmp_path, 'clot', 'first.tiff', 2)
    with pytest.raises(ValueError, match='missing'):
        learning.training_request(tmp_path, 'clot', base_version=model['id'])


def test_deleting_all_versions_keeps_review_library_and_restores_choice(tmp_path):
    record = submission(tmp_path, 'clot', 'first.tiff', 1)
    version = learning.train(learning.training_request(tmp_path, 'clot'), lambda *a: None)['version']['id']
    learning.manage_model(tmp_path, 'clot', 'delete_model', version)
    state = learning.manage_model(tmp_path, 'clot', 'delete_model', version)
    assert state['selected_version'] == '' and state['versions'] == [] and len(state['deleted_versions']) == 1
    assert learning.training_request(tmp_path, 'clot')['examples'][0]['id'] == record['id']
    with pytest.raises(ValueError, match='deleted'):
        learning.manage_model(tmp_path, 'clot', 'select_model', version)
    with pytest.raises(ValueError, match='no longer available'):
        learning.manage_model(tmp_path, 'clot', 'restore_model', '../outside')


def test_shared_quality_preserves_blur_models_and_archived_examples(tmp_path):
    first = submission(tmp_path, 'blur', 'first.tiff', 11)
    old = learning.train(learning.training_request(tmp_path, 'blur'), lambda *a: None)['version']
    origin = learning.folder(tmp_path, 'blur')
    files = {p: p.read_bytes() for p in origin.glob('*') if p.is_file()}
    ref = 'blur:' + old['id']
    combined = learning.state(tmp_path, 'debris')
    assert combined['selected_version'] == ref
    assert combined['versions'][0]['source_part'] == 'blur'
    assert learning.manage_model(tmp_path, 'debris', 'select_model', ref)['selected_version'] == ref
    assert learning.model(tmp_path, 'debris', ref)[1]['id'] == old['id']
    extra = submission(tmp_path, 'debris', 'second.tiff', 21)
    unrelated = submission(tmp_path, 'blur', 'later.tiff', 12)
    request = learning.training_request(tmp_path, 'debris', base_version=ref, current_snapshot=extra['id'])
    assert [(e['id'], e['source_part']) for e in request['examples']] == [(first['id'], 'blur'), (extra['id'], 'debris')]
    child = learning.train(request, lambda *a: None)['version']
    assert child['parent_version'] == ref and child['artifact_targets'] == ['blur', 'bright', 'lint']
    assert child['example_sources'] == {first['id']: 'blur', extra['id']: 'debris'}
    restored = learning.training_request(tmp_path, 'debris', base_version=child['id'])
    assert restored['examples'] == request['examples']
    assert unrelated['id'] not in child['examples']
    # Read/select/branch never change the original version or its saved samples.
    assert all(p.read_bytes() == value for p, value in files.items() if p.name != 'examples.json')
    deleted = learning.manage_model(tmp_path, 'debris', 'delete_model', ref)
    assert ref in {v['id'] for v in deleted['deleted_versions']}
    assert child['id'] in {v['id'] for v in deleted['versions']}
    # Deleting the parent only hides the model; a child still owns these samples.
    assert learning.training_request(tmp_path, 'debris', base_version=child['id'])['examples'] == restored['examples']
    with pytest.raises(ValueError, match='saved model'):
        learning.model(tmp_path, 'debris', ref)
    assert learning.manage_model(tmp_path, 'debris', 'restore_model', ref)['selected_version'] == ref
    (origin / (first['id'] + '.npz')).unlink()
    with pytest.raises(ValueError, match='missing'):
        learning.training_request(tmp_path, 'debris', base_version=child['id'])
    with pytest.raises(ValueError, match='saved model'):
        learning.model(tmp_path, 'clot', ref)
    with pytest.raises(ValueError, match='saved model'):
        learning.model(tmp_path, 'debris', 'clot:arbitrary')
