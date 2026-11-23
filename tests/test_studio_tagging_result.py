"""Completed tagging links to current geometry; unrelated or mismatched results do not."""
from pathlib import Path

import pandas as pd
import pytest

from spacemap_gui.studio import tagging
from spacemap_gui.tabs import cell_tagging_ui as tab5
from spacemap_gui.viewer import registry
from test_studio_commands import setup  # noqa: F401


@pytest.fixture
def saved_tags(setup, monkeypatch):
    channel, source, root, _ = setup
    token = registry.session_view(channel)['spec']['layers'][1]['token']
    folder = root / 'tags'
    folder.mkdir()
    state = {tab5.OUTPUT_DIR_KEY: str(folder), tab5.OUTPUT_SUFFIX_KEY: '_custom_tags'}
    found = {'mode': tab5.MODE_PER_IMAGE,
             'rows': [{'source': 'P11_roi_2_cell_features.csv', 'image': 'P11_roi_2.ome.tiff'}]}
    monkeypatch.setattr(tagging, 'values', lambda *args: state)
    monkeypatch.setattr(tagging, 'sources', lambda *args: found)
    frame = pd.read_csv(source).query("batch_id == 'P11_roi_2'").drop(columns='batch_id')
    frame['tissue'] = 1
    path = folder / 'P11_roi_2_cell_features_custom_tags.csv'
    return root, token, frame, path, state, found


def test_published_tags_link_without_reopening_or_copying(saved_tags):
    root, token, frame, path, state, found = saved_tags
    # A staging file is not a completed result.
    frame.to_csv(path.with_name('.staged.csv'), index=False)
    assert tagging.output_versions(state, found) == {}
    assert not tagging.link_output(root, token)['linked']
    frame.to_csv(path, index=False)
    original = registry.resolve_object(token, 'cells')
    link = tagging.link_output(root, token, [str(path)])
    assert link['linked'] and link['table'].startswith('tagged_') and 'tissue' in link['columns']
    current = registry.resolve_object(token, 'cells')
    assert current['image'] == original['image'] and current['roi'] == original['roi']
    assert Path(current['tables'][link['table']]) == path
    assert current['tables']['t0'] == original['tables']['t0']
    assert tagging.link_output(root, token, [str(path)], link['version'])['unchanged']
    frame['tissue'] = 22
    frame.to_csv(path, index=False)
    assert tagging.link_output(root, token, [str(path)], link['version'])['linked']


def test_other_images_and_changed_segmentation_do_not_replace_current_cells(saved_tags):
    root, token, frame, path, _, _ = saved_tags
    frame.to_csv(path, index=False)
    before = dict(registry.resolve_object(token, 'cells')['tables'])
    assert not tagging.link_output(root, token, [str(path.parent / 'other.csv')])['linked']
    frame['x_centroid'] += 50
    frame.to_csv(path, index=False)
    answer = tagging.link_output(root, token, [str(path)])
    assert not answer['linked'] and 'another segmentation' in answer['reason']
    assert registry.resolve_object(token, 'cells')['tables'] == before


def test_corrected_image_alias_matches_only_one_verified_roi(saved_tags):
    root, token, frame, path, _, found = saved_tags
    frame.to_csv(path, index=False)
    row = found['rows'][0]
    row.update(image='P11_roi_2_bg_subtracted.ome.tiff', sample='P11_roi_2')
    assert tagging.link_output(root, token, [str(path)])['linked']
    found['rows'].append({**row, 'source':'another_P11_roi_2.csv'})
    assert not tagging.link_output(root, token, [str(path)])['linked']
