"""Scientific denominators, repeated patients, and saved figure reproducibility."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import marker_exploration as core
from spacemap_gui.studio import marker_figures as service, subphenotypes


def test_denominators_missing_values_and_equal_patient_weight():
    frame = pd.DataFrame({
        'phenotype': ['A'] * 6 + ['B'] * 2,
        'M_corr': [0, 0, 0, 0, 10, np.nan, 10, 10],
        'patient': ['large'] * 4 + ['small'] * 2 + ['large', 'small'],
    })
    b = core.summarize(frame, 'phenotype', ['M_corr'], patient='patient', z=0, min_cells=1)
    s = b['summary'].set_index('phenotype')
    assert b['thresholds'].iloc[0].cutoff == pytest.approx(30 / 7)
    assert s.loc['A', 'pct_high'] == 20 and s.loc['B', 'pct_high'] == 100
    assert s.loc['A', 'composition_pct'] == pytest.approx(100 / 3)
    assert s.loc['B', 'composition_pct'] == pytest.approx(200 / 3)
    h = b['histograms'].iloc[0]
    assert h.counts[0] == 4 and h.counts[-1] == 1
    assert h.balanced[0] == 50 and h.balanced[-1] == 50
    assert s.loc['A', 'n_cells'] == 6 and s.loc['A', 'n_finite'] == 5
    b = core.summarize(frame, 'phenotype', ['M_corr'], patient='patient', z=0, min_cells=2)
    assert b['histograms'].iloc[0].balanced[0] == 100  # one finite cell is not an eligible patient


def test_raw_is_transformed_once_corrected_is_kept_and_empty_is_not_zero():
    f = pd.DataFrame({'p': ['A'] * 4, 'raw': [0, 5, 10, np.inf], 'M_corr': [0, 1, 2, np.nan],
                      'constant': [4] * 4, 'missing': [np.nan] * 4})
    b = core.summarize(f, 'p', list(f.columns[1:]))
    t = b['thresholds'].set_index('marker')
    assert t.loc['raw', 'mean'] == pytest.approx(np.arcsinh(np.array([0, 5, 10]) / 5).mean())
    assert t.loc['M_corr', 'mean'] == 1
    s = b['summary'].set_index('marker')
    assert s.loc['constant', 'pct_high'] == 0 and s.loc['constant', 'iqr'] == 0
    assert np.isnan(s.loc['missing', 'pct_high'])
    assert np.isnan(s.loc['constant', 'composition_pct'])
    with core.plot_style():
        for kind in ['dot', 'composition', 'histogram', 'variation']:
            fig = core.draw(b, kind, phenotype='A')
            fig.canvas.draw()
            fig.clear()
    stored = core.summarize(f, 'p', ['raw'], transform='stored')
    assert stored['thresholds'].iloc[0]['mean'] == 5


def test_strata_keep_cutoffs_and_separate_repeated_patients():
    # Patient p1 contributes many low cells at baseline and high cells at followup.
    # Patient p2 contributes one high baseline cell: a pooled fraction would be 20%, not 50%.
    f = pd.DataFrame({'p': ['A'] * 9, 'patient': ['p1'] * 4 + ['p2'] + ['p1'] * 2 + ['p2', 'p3'],
        'timepoint': ['before'] * 5 + ['after'] * 3 + [None], 'arm': ['treated'] * 9,
        'M_corr': [0] * 4 + [10] * 5})
    b = core.summarize(f, 'p', ['M_corr'], patient='patient', stratify=['arm', 'timepoint'], z=0, min_cells=1)
    assert b['thresholds'].iloc[0].cutoff == pytest.approx(50 / 9)
    assert b['metadata']['missing_stratum_cells'] == 1
    assert b['summary'].iloc[0].n_finite == 9  # missing group does not erase overall measurements
    groups = b['groups'].set_index('stratum')
    before = next(s for s, label in b['metadata']['strata'].items() if 'before' in label)
    after = next(s for s, label in b['metadata']['strata'].items() if 'after' in label)
    assert groups.loc[before, 'pct_high'] == 50
    assert groups.loc[after, 'pct_high'] == 100
    assert groups.loc[before, 'n_eligible'] == 2
    repeated = b['stratified'].query("patient == 'p1'").set_index('stratum')
    assert repeated.loc[before, 'pct_high'] == 0 and repeated.loc[after, 'pct_high'] == 100
    pooled = core.summarize(f, 'p', ['M_corr'], stratify=['timepoint'], z=0, min_cells=1)
    before = next(s for s, label in pooled['metadata']['strata'].items() if 'before' in label)
    assert pooled['groups'].set_index('stratum').loc[before, 'pct_high'] == 20
    with core.plot_style():
        for kind in ['stratified', 'groups']:
            fig = core.draw(b, kind, phenotype='A')
            fig.canvas.draw()
            fig.clear()


@pytest.fixture
def generated(tmp_path, monkeypatch):
    monkeypatch.setenv('SPACEMAP_FAST_COPY', 'off')
    table = tmp_path / 'cells.parquet'
    f = pd.DataFrame({'phenotype_test': ['Epithelial'] * 24 + ['Others'] * 6,
        'patient': ['p1', 'p2'] * 15, 'timepoint': ['pre'] * 12 + ['post'] * 18,
        'region': [1] * 20 + [0] * 10, 'A_corr': np.arange(30, dtype=float),
        'B_corr': np.tile([1., 5., 2.], 10)})
    f.loc[3, 'B_corr'] = np.nan
    f.to_parquet(table, index=False)
    config = {**subphenotypes.DEFAULTS, 'parent_col': 'phenotype_test', 'parent_values': ['Epithelial'],
              'markers': ['A_corr', 'B_corr'], 'filters': [{'column': 'region', 'op': 'in', 'values': ['1']}],
              'figure_options': {'patient_column': 'patient', 'stratify': ['timepoint'], 'min_cells': 1, 'z': 0,
                                 'figures': list(service.FIGURES)}}
    before = table.read_bytes()
    settings = service.prepare(tmp_path, table, config)
    result = service.run(settings, tmp_path, lambda *args: None)
    assert table.read_bytes() == before
    return tmp_path, table, settings, result


def test_exports_views_keep_source_and_thresholds(generated, monkeypatch):
    from spacemap_gui.studio import jobs
    root, table, settings, result = generated
    output = Path(result['output_dir'])
    assert result['metadata']['cells'] == 20 and result['metadata']['phenotypes'] == ['Epithelial']
    assert {e['kind'] for e in result['figures']} == {'review', 'dot', 'composition', 'histogram', 'stratified', 'groups', 'variation', 'coexpression'}
    assert set(service.TABLES) <= {p.stem for p in output.glob('*.parquet')}
    assert (output / 'figure_book.pdf').read_bytes().startswith(b'%PDF')
    for entry in result['figures']:
        assert (output / (entry['id'] + '.png')).read_bytes().startswith(b'\x89PNG')
        assert (output / (entry['id'] + '.pdf')).stat().st_size > 1000
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['selection']['filters'][0]['column'] == 'region'
    assert metadata['stratify'] == ['timepoint']
    monkeypatch.setattr(jobs, 'status', lambda *args: {'kind': 'marker_figures', 'status': 'done', 'result': result})
    threshold_bytes = (output / 'thresholds.parquet').read_bytes()
    histogram = next(e for e in result['figures'] if e['kind'] == 'histogram')
    view = service.render(root, 'test', {'figure': histogram['id'], 'markers': ['B_corr']})
    assert view['markers'] == ['B_corr']
    assert service.figure_file(root, 'test', view['file'] + '.png').is_file()
    assert (output / 'thresholds.parquet').read_bytes() == threshold_bytes
    assert service.render(root, 'test', {'figure': histogram['id'], 'markers': ['B_corr']}) == view
    styled = service.render(root, 'test', {'figure': histogram['id'], 'markers': ['B_corr'],
        'appearance': {'font_size': 12, 'x_size': 8, 'x_angle': 45, 'width': 140}})
    assert styled['file'] != view['file']
    saved_style = json.loads((output / (styled['file'] + '.json')).read_text())
    assert saved_style['appearance']['x_angle'] == 45 and saved_style['appearance']['width'] == 140
    assert (output / (styled['file'] + '.png')).read_bytes() != (output / (view['file'] + '.png')).read_bytes()
    assert service.figure_file(root, 'test', styled['file'] + '.pdf').stat().st_size > 1000
    assert (output / 'thresholds.parquet').read_bytes() == threshold_bytes
    with pytest.raises(ValueError):
        service.figure_file(root, 'test', '../cells.parquet')
    with pytest.raises(ValueError):
        service.render(root, 'test', {'figure': histogram['id'], 'markers': ['bogus']})
    with pytest.raises(ValueError):
        service.render(root, 'test', {'figure': histogram['id'], 'markers': []})
    with pytest.raises(ValueError, match='appearance'):
        service.render(root, 'test', {'figure': histogram['id'], 'appearance': {'unsupported': 1}})
    # A result remains usable after edits to the source because it uses its saved measurements.
    pd.DataFrame({'changed': [1]}).to_parquet(table)
    dot = next(e for e in result['figures'] if e['kind'] == 'dot')
    assert service.render(root, 'test', {'figure': dot['id'], 'markers': ['A_corr']})['file']
    with pytest.raises(ValueError, match='table changed'):
        service.run(settings, root, lambda *args: None)


def test_options_and_settings_persist_without_affecting_cluster_inputs(tmp_path):
    from spacemap_gui.studio.settings import revision
    saved = subphenotypes.values(tmp_path)
    options = {'patient_column': 'patient', 'stratify': ['arm', 'response'], 'z': 2, 'variation': 'iqr',
               'figures': ['histogram', 'patients']}
    result = subphenotypes.save(tmp_path, {'figure_options': options}, revision(saved))
    assert result['figure_options'] == options
    assert result['markers'] == saved['markers'] and result['parent_col'] == saved['parent_col']
    for invalid in [{'z': float('nan')}, {'cofactor': 0}, {'min_cells': 2.5},
                    {'stratify': ['patient'], 'patient_column': 'patient'}, {'pair': ['a', 'a']},
                    {'figures': ['bogus']}, {'figures': 'dot'}, {'figures': ['dot', 'dot']},
                    {'figures': [{}]}]:
        with pytest.raises(ValueError):
            service.options(invalid)


def test_single_marker_without_patients_and_missing_strata_are_supported():
    f = pd.DataFrame({'p': ['A', 'A'], 'M_corr': [1., 2.], 'response': [None, None], 'patient': [None, None]})
    b = core.summarize(f, 'p', ['M_corr'], patient='patient', stratify=['response'])
    assert {e['kind'] for e in service.recipes(b, 'iqr')} == {'review', 'dot', 'composition', 'histogram', 'variation'}


def test_dot_only_exports_and_skips_unselected_calculations(tmp_path, monkeypatch):
    def unwanted(*args, **kwargs):
        pytest.fail('An unselected distribution was calculated.')
    monkeypatch.setattr(np, 'histogram', unwanted)
    monkeypatch.setattr(np, 'histogram2d', unwanted)
    table = tmp_path / 'cells.parquet'
    pd.DataFrame({'p': ['A'] * 4, 'M_corr': [1., 2., 3., 4.]}).to_parquet(table, index=False)
    config = {**subphenotypes.DEFAULTS, 'parent_col': 'p', 'parent_values': ['A'], 'markers': ['M_corr'],
        # Hidden preferences from another view/table must not block this run.
        'figure_options': {'patient_column': 'old_patient', 'stratify': ['old_arm'],
                           'pair': ['old_x', 'old_y'], 'figures': ['dot']}}
    settings = service.prepare(tmp_path, table, config)
    result = service.run(settings, tmp_path, lambda *args: None)
    output = Path(result['output_dir'])
    assert [e['kind'] for e in result['figures']] == ['dot']
    assert {p.name for p in output.glob('*.png')} == {'001_dot.png'}
    assert {p.name for p in output.glob('*.pdf')} == {'001_dot.pdf', 'figure_book.pdf'}
    assert result['metadata']['options']['figures'] == ['dot']
    assert result['metadata']['options']['patient_column'] == 'old_patient'
    assert result['metadata']['patient_column'] == ''
    assert result['metadata']['stratify'] == [] and result['metadata']['pair'] == []
    for name in ['histograms', 'patients', 'stratified', 'groups', 'coexpression']:
        assert pd.read_parquet(output / (name + '.parquet')).empty
    assert pd.read_parquet(output / 'summary.parquet').iloc[0]['median'] == 2.5


@pytest.mark.parametrize('kinds,expected', [
    (['histogram'], {'histogram'}), (['patients'], {'stratified'}),
    (['groups'], {'groups'}), (['variation', 'composition'], {'variation', 'composition'}),
    (['coexpression'], {'coexpression'}),
])
def test_selected_recipes_only(kinds, expected):
    frame = pd.DataFrame({'p': ['A', 'A', 'B', 'B'], 'patient': ['1', '2', '1', '2'],
                          'arm': ['x'] * 4, 'M_corr': [1., 2., 3., 4.], 'N_corr': [2., 4., 1., 3.]})
    bundle = core.summarize(frame, 'p', ['M_corr', 'N_corr'], patient='patient', stratify=['arm'],
                            pair=['M_corr', 'N_corr'], kinds=kinds, min_cells=1)
    assert {e['kind'] for e in service.recipes(bundle, 'iqr', kinds)} == expected
    if 'histogram' not in kinds:
        assert bundle['histograms'].empty
    if 'groups' not in kinds:
        assert bundle['groups'].empty
    if 'coexpression' not in kinds:
        assert bundle['coexpression'].empty


def test_selected_figure_requirements(tmp_path):
    table = tmp_path / 'cells.parquet'
    pd.DataFrame({'p': ['A', 'A'], 'M_corr': [1., 2.], 'patient': [None, None]}).to_parquet(table)
    config = {**subphenotypes.DEFAULTS, 'parent_col': 'p', 'parent_values': ['A'], 'markers': ['M_corr']}
    assert service.prepare(tmp_path, table, config)['options']['figures'] == ['dot']
    assert service.options({'figures': []})['figures'] == []  # Saving an unfinished selection is allowed.
    for prefs, message in [
        ({'figures': []}, 'Select at least one'),
        ({'figures': ['patients']}, 'patient column'),
        ({'figures': ['groups']}, 'Stratify by'),
        ({'figures': ['coexpression']}, 'two different markers'),
    ]:
        with pytest.raises(ValueError, match=message):
            service.prepare(tmp_path, table, {**config, 'figure_options': prefs})
    settings = service.prepare(tmp_path, table, {**config,
        'figure_options': {'figures': ['patients'], 'patient_column': 'patient'}})
    with pytest.raises(ValueError, match='No figures to draw'):
        service.run(settings, tmp_path, lambda *args: None)


@pytest.mark.parametrize('style', [
    {'font_size': float('nan')}, {'width': float('inf')}, {'height': -1},
    {'x_angle': 100}, {'show_values': 'yes'}, {'wrap': 2.5}, {'label_pad': True}, {'x_size': None},
])
def test_invalid_appearance_is_rejected(style):
    with pytest.raises(ValueError):
        core.appearance_options(style)


def test_heatmap_labels_fit_and_appearance_preserves_measurements(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    f = pd.DataFrame({'p': ['Epithelial'] * 16, 'patient': [str(i) for i in range(1, 17)],
        'arm': [1] * 16, 'timepoint': [1] * 16, 'response': ['SD', 'CR'] * 8,
        'Ki67_corr': np.arange(16, dtype=float)})
    bundle = core.summarize(f, 'p', ['Ki67_corr'], patient='patient', stratify=['arm', 'timepoint', 'response'],
                            kinds=['patients'], min_cells=1)
    original = bundle['stratified'].copy(deep=True)
    with core.plot_style():
        for style in [{}, {'font_size': 12, 'x_size': 8, 'y_size': 11, 'x_angle': 45,
                           'width': 130, 'label_pad': 10, 'show_values': False}]:
            fig = core.draw(bundle, 'stratified', appearance=style)
            canvas = FigureCanvasAgg(fig); canvas.draw()
            renderer = canvas.get_renderer()
            ax = fig.axes[0]
            boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
            assert all(a.x1 < b.x0 for a, b in zip(boxes, boxes[1:])), 'Adjacent group labels overlap'
            footer = fig.texts[-1].get_window_extent(renderer)
            assert min(b.y0 for b in boxes) > footer.y1, 'Group labels overlap the caption'
            assert all(b.x0 >= 0 and b.x1 <= fig.bbox.width and b.y0 >= 0 for b in boxes)
            if style:
                assert ax.get_xticklabels()[0].get_fontsize() == 8
                assert ax.get_xticklabels()[0].get_rotation() == 45
                assert not any(text.get_visible() for text in ax.texts)
            fig.savefig(tmp_path / ('custom.png' if style else 'automatic.png'), dpi=100)
            fig.clear()
    pd.testing.assert_frame_equal(bundle['stratified'], original)
