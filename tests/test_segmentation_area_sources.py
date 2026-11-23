"""A segmentation area belongs to one GeoJSON, even when names are repeated."""
import json

import pytest

from spacemap_gui.studio import annotation_inputs, segmentation
from spacemap_gui.studio.settings import read_config
from test_studio_commands import setup, run  # noqa: F401
from test_studio_segmentation import _square


def sources(root):
    image = root / 'outputs/rois/P11_roi_2.ome.tiff'
    folder = annotation_inputs.import_folder(root, image)
    folder.mkdir(parents=True)
    files = []
    for filename, positions in [('aaaa_review.geojson', [0, 30]),
                                 ('bbbb_merged.geojson', [0, 60]),
                                 ('cccc_merged.geojson', [0, 90])]:
        path = folder / filename
        path.write_text(json.dumps({'type': 'FeatureCollection',
            'features': [_square(str(x), x) for x in positions]}))
        files.append(path)
    return image, files


def test_duplicate_names_require_a_file_and_retain_each_files_pieces(setup):
    root = setup[2]
    image, files = sources(root)
    saved = segmentation.saved_annotations(root, read_config(root), {}, [image.name])[image.name]
    assert saved['imported:tissue']['ambiguous']
    assert saved['merged › tissue']['ambiguous']  # two copies with the same name
    specific = {key: value for key, value in saved.items() if key.startswith('source:')}
    assert len(specific) == 3
    assert {item['path'] for item in specific.values()} == {str(p) for p in files}
    groups = segmentation._area_groups(saved, 'imported:tissue')
    assert len(groups) == 3 and len({label for label, _ in groups}) == 3
    assert all(label.startswith('File · ') for label, _ in groups)
    chosen = next(key for key, item in specific.items() if item['path'] == str(files[0]))
    segmentation.save_plan(root, {'areas': {image.name: chosen}})
    row = segmentation.plan(root, read_config(root), {})['rows'][0]
    assert row['area'] == chosen and not row['ambiguous'] and not row['missing']
    assert row['area_source'] == str(files[0])
    mask = segmentation.area_path(root, read_config(root), {}, image.name, specific[chosen])
    features = json.loads(open(mask).read())['features']
    assert len(features) == 2  # all pieces of the selected file, no other imports
    assert {f['geometry']['coordinates'][0][0][0] for f in features} == {0, 30}


def test_ambiguous_saved_area_fails_before_previous_results_are_moved(setup, monkeypatch):
    from spacemap_gui.studio import jobs
    root = setup[2]
    image, _ = sources(root)
    segmentation.save_plan(root, {'areas': {image.name: 'imported:tissue'}})
    row = segmentation.plan(root, read_config(root), {})['rows'][0]
    assert row['ambiguous'] and not row['missing']
    table, _ = segmentation.results(root / 'outputs/segmentation', image.name)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text('label,x\n1,20\n')
    started = []
    monkeypatch.setattr(jobs, 'start', lambda *a, **kw: started.append(kw))
    result = run(setup, 'segment_images', {'again': True, 'tag': False}, confirmed=True)
    assert result['status'] != 'done' and 'Choose one file' in json.dumps(result)
    assert not started and table.read_text() == 'label,x\n1,20\n'
    assert not (table.parent / 'previous').exists()


def test_worker_logs_full_source_annotation_and_resolved_mask(setup, monkeypatch, capsys):
    from spacemap_gui.studio import tasks
    root = setup[2]
    image, files = sources(root)
    saved = segmentation.saved_annotations(root, read_config(root), {}, [image.name])[image.name]
    area = next(value for key, value in saved.items() if key.startswith('source:') and value['path'] == str(files[0]))
    captured = []
    monkeypatch.setattr(segmentation, 'run', lambda items, *args: captured.extend(items) or [{'cells': 2}])
    tasks.segment_images({'root': str(root), 'values': {}, 'images': [
        {'image': str(image), 'area': {'name': 'tissue', 'pending': area}}]}, root, lambda *args: None)
    log = capsys.readouterr().out
    assert f'Segmentation source GeoJSON: {files[0]}' in log
    assert 'Selected annotation: tissue' in log
    assert f"Segmentation mask GeoJSON: {captured[0]['area']['path']}" in log
    assert str(files[1]) not in log and str(files[2]) not in log
    config = segmentation.config_for(segmentation.values(root, read_config(root)))
    notes = segmentation.region(config, image.name, segmentation.values(root, read_config(root)), captured[0]['area'])
    assert captured[0]['area']['path'] in notes[-1]
    # A later edit or another run cannot replace this job's exact area.
    snapshot = captured[0]['area']['path']
    before = open(snapshot).read()
    files[0].write_text(json.dumps({'type': 'FeatureCollection', 'features': []}))
    assert open(snapshot).read() == before and snapshot != str(files[0])


def test_worker_rejects_a_legacy_request_that_combined_sources(setup, monkeypatch):
    from spacemap_gui.studio import tasks
    root = setup[2]
    image, files = sources(root)
    ran = []
    monkeypatch.setattr(segmentation, 'run', lambda *args: ran.append(True))
    area = {'path': str(files[0]), 'paths': [str(p) for p in files], 'name': 'tissue'}
    with pytest.raises(ValueError, match='Choose one file'):
        tasks.segment_images({'root': str(root), 'values': {}, 'images': [
            {'image': str(image), 'area': {'name': 'tissue', 'pending': area}}]}, root, lambda *args: None)
    assert not ran and not (root / 'outputs/segmentation/areas').exists()
