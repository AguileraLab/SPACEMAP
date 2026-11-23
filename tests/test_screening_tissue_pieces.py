"""Separate tissue cores are measured first; people remain the inference unit."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.studio import screening as s
from test_studio_screening import design


@pytest.mark.parametrize('aggregation', ['median', 'mean'])
def test_core_values_are_summarized_within_visit_before_comparing_changes(aggregation):
    rows = []
    for person in range(6):
        for visit, positives in [('Before', [0, 4, 24]), ('After', [1, 8, 27])]:
            for core, (size, positive) in enumerate(zip([10, 20, 30], positives), 1):
                for cell in range(size):
                    rows.append(dict(subject=str(person), outcome='Control' if person < 3 else 'Case',
                        image=f'{person}_{visit}', visit=visit, core=core, x=cell, y=core,
                        type='A' if cell < positive else 'B', distance=1))
    frame = pd.DataFrame(rows)
    c = design(region_col='core', aggregation=aggregation, widths=[5], phenotypes=['A'])
    c['endpoint'].update(change_col='visit', change_before='Before', change_after='After')
    result, samples, regions, audit, _ = s.compute(frame, c, dict.fromkeys(frame.image.unique(), .5))
    summarize = np.median if aggregation == 'median' else np.mean
    expected = summarize([10, 40, 90]) - summarize([0, 20, 80])
    np.testing.assert_allclose(samples.value, expected)
    assert len(regions) == 36  # six people, two visits, three cores
    assert len(samples) == audit['independent_samples'] == 6
    assert result.n_samples.iloc[0] == 6
    for visit, expected_core_values in [('Before', [0, 20, 80]), ('After', [10, 40, 90])]:
        measured = regions[(regions._sample == '0') & (regions._occasion == visit)].sort_values('_region')
        np.testing.assert_allclose(measured.value, expected_core_values)
    # The plain pooled comparison deliberately summarizes all eligible cores
    # from both visits. It must not be mistaken for a visit-specific comparison.
    pooled = deepcopy(c)
    pooled['endpoint'].update(change_col='', change_before='', change_after='')
    _, pooled_samples, _, _, _ = s.compute(frame, pooled, dict.fromkeys(frame.image.unique(), .5))
    np.testing.assert_allclose(pooled_samples.value, summarize([0, 20, 80, 10, 40, 90]))


def test_neighbours_pairs_and_triangles_never_cross_core_ids():
    rows = []
    for person in range(6):
        for core, labels in [(1, ['A', 'A', 'B']), (2, ['B', 'C', 'C'])]:
            for cell, label in enumerate(labels):
                rows.append(dict(subject=str(person), outcome='Control' if person < 3 else 'Case',
                    image=str(person), core=core, x=cell + core / 10, y=0,
                    type=label, distance=1))
    frame = pd.DataFrame(rows)
    c = design(region_col='core', widths=[5], families=['nn_distance', 'dyad', 'triad'])
    for name in ('dyad', 'triad'):
        c['options'][name].update(metric='count', max_um=30)
    _, _, separated, _, _ = s.compute(frame, c, dict.fromkeys(frame.image.unique(), .5))
    assert (separated.loc[(separated.family == 'dyad') & (separated.label == 'A / C'), 'value'] == 0).all()
    assert (separated.loc[separated.family == 'triad', 'value'] == 0).all()
    assert separated.loc[(separated.family == 'nn_distance') & (separated.label == 'A / C'), 'value'].isna().all()
    c['region_col'] = ''
    _, _, joined, _, _ = s.compute(frame, c, dict.fromkeys(frame.image.unique(), .5))
    assert (joined.loc[(joined.family == 'dyad') & (joined.label == 'A / C'), 'value'] > 0).all()
    assert (joined.loc[joined.family == 'triad', 'value'] > 0).all()
