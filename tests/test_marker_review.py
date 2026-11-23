"""Universal review, exact selected-cell membership and stale-result protection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import marker_exploration, marker_review
from spacemap_gui.studio import marker_figures, marker_review as service, subphenotypes, flow_gating
from spacemap_gui.studio.settings import revision
from test_studio_commands import setup  # noqa: F401


def test_candidates_are_descriptive_and_groups_are_optional(tmp_path):
    f = pd.DataFrame({'kind': ['A'] * 12, 'signal': range(12), 'flat': [3.] * 12,
                      'missing': [np.nan] * 12, 'specimen_code': ['X'] * 6 + ['Y'] * 6})
    for grouping in ['', 'specimen_code']:
        b = marker_exploration.summarize(f, 'kind', ['signal', 'flat', 'missing'],
            patient=grouping, min_cells=2, kinds=['histogram'], transform='stored')
        rows = marker_review.records(b, 'A')
        assert [r['marker'] for r in rows if r['candidate']] == ['signal']
        assert rows[0]['iqr'] == 5.5 and rows[0]['coverage'] == 100
        assert b['metadata']['patient_column'] == grouping
        with marker_exploration.plot_style():
            fig = marker_review.draw(b, 'A')
            fig.savefig(tmp_path / ('review_' + (grouping or 'none') + '.png'))
            fig.clear()


def make_review(root, table, monkeypatch, **extra):
    config = subphenotypes.save(root, {
        'parent_col': 'phenotype_ml_consensus', 'parent_values': ['CD8', 'B'],
        'markers': ['M0', 'M1'], 'transform': 'auto', 'cofactor': 5.,
        'figure_options': {'figures': ['review'], 'min_cells': 1}, **extra},
        revision(subphenotypes.values(root)))
    settings = marker_figures.prepare(root, table, config, subphenotypes.image_scope(root, table))
    result = marker_figures.run(settings, root, lambda *a: None)
    monkeypatch.setattr(marker_figures, 'saved_result', lambda *_: result)
    return config, result


def test_selected_rows_match_data_and_pixels_and_only_explicit_choices_apply(setup, monkeypatch):
    channel, table, root, _ = setup
    before = table.read_bytes()
    config, result = make_review(root, table, monkeypatch,
        filters=[{'column': 'label', 'op': 'not_in', 'values': ['1']}])
    body = {'job_id': 'test', 'phenotype': 'CD8'}
    evidence = service.evidence(root, body)
    assert evidence['grouping'] == '' and evidence['samples'] == []
    assert evidence['rows'][0]['n_cells'] == 1  # Other image and label 1 excluded.
    answer = service.select(root, channel, table, {**body, 'marker': 'M0', 'lo': .38, 'hi': .4})
    token = answer['selection']['token']
    assert answer['selection']['count'] == 1
    assert flow_gating.token_rows(token, channel, table).tolist() == [1]
    indices, lut = flow_gating.tile(token, str(table), 'P11_roi_2', np.array([[1, 2, 3, 4]], dtype=np.int64))
    assert (lut[indices][0, :, 3] > 0).tolist() == [False, True, False, False]
    assert table.read_bytes() == before
    assert subphenotypes.values(root)['chosen_markers'] == {}
    applied = service.apply(root, table, {**body, 'markers': ['M1'], 'revision': revision(config)})
    assert applied['chosen_markers'] == {'CD8': ['M1']}
    changed = subphenotypes.values(root)
    subphenotypes.save(root, {'cofactor': 10.}, revision(changed))
    with pytest.raises(ValueError, match='settings changed'):
        service.apply(root, table, {**body, 'markers': ['M0'], 'revision': revision(subphenotypes.values(root))})
    with table.open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='changed'):
        service.select(root, channel, table, {**body, 'marker': 'M0'})


def test_review_with_custom_group_is_optional_and_saved(setup, monkeypatch):
    _, table, root, _ = setup
    config, result = make_review(root, table, monkeypatch,
        figure_options={'figures': ['review'], 'patient_column': 'batch_id', 'min_cells': 1})
    evidence = service.evidence(root, {'job_id': 'test', 'phenotype': 'CD8'})
    assert evidence['grouping'] == 'batch_id'
    assert {r['patient'] for r in evidence['samples']} == {'P11_roi_2'}
    assert {e['kind'] for e in result['figures']} == {'review'}
    assert result['metadata']['cells'] == 4
    for bounds in [{'lo': 2, 'hi': 1}, {'lo': float('nan')}]:
        with pytest.raises(ValueError):
            service.select(root, setup[0], table, {'job_id': 'test', 'phenotype': 'CD8', 'marker': 'M0', **bounds})
