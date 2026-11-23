"""Descriptive marker review. A shortlist is an inspection priority, never a gate."""
import json
import numpy as np


RULE = ('Inspection order: mean rank of IQR/reference SD and 5–95% spread/reference SD, '
        'within each phenotype. Reference SD uses all selected cells before clustering normalization. '
        'Up to six variable markers with enough measured cells are suggested for inspection. '
        'This describes relative spread, not signal above background or biological importance. '
        'Rare states can be missed. Sample differences may be biological or technical.')


def records(bundle, phenotype):
    """Use exact saved summaries, without resampling, fitting gates or changing labels."""
    rows = bundle['summary'].query('phenotype == @phenotype').copy()
    ref = bundle['thresholds'].set_index('marker').sd
    rows['tail'] = rows.p95 - rows.p05
    rows['relative_tail'] = rows['tail'] / rows.marker.map(ref).replace(0, np.nan)
    rows['coverage'] = 100 * rows.n_finite / rows.n_cells
    eligible = (rows.n_finite >= max(2, bundle['metadata']['min_cells'])) & (rows.sd > 0)
    rows['priority'] = np.nan
    rows.loc[eligible, 'priority'] = rows.loc[eligible, ['relative_iqr', 'relative_tail']].rank(
        ascending=False, method='min').mean(axis=1)
    rows = rows.sort_values(['priority', 'marker'], na_position='last')
    rows['candidate'] = False
    rows.loc[rows.index[rows.priority.notna()][:6], 'candidate'] = True
    rows['note'] = np.where(rows.n_finite < max(2, bundle['metadata']['min_cells']),
                            'Too few measured cells', np.where(rows.sd.eq(0), 'Constant', 'Inspect staining'))
    return json.loads(rows.to_json(orient='records'))


def draw(bundle, phenotype, markers=None, **_):
    """A shareable overview; the interactive review exposes every selected marker."""
    from matplotlib.figure import Figure
    from .marker_exploration import display_name
    rows = records(bundle, phenotype)
    if markers:
        rows = [r for r in rows if r['marker'] in markers]
    rows = rows[:6]
    fig = Figure(figsize=(14, 2.1 * max(1, len(rows)) + 2.1), layout='constrained')
    axes = fig.subplots(max(1, len(rows)), 3, squeeze=False)
    for i, row in enumerate(rows):
        marker = row['marker']
        hist = bundle['histograms'].query('phenotype == @phenotype and marker == @marker')
        ax = axes[i, 0]
        if len(hist):
            h = hist.iloc[0]
            counts = np.asarray(h.counts, dtype=float)
            ax.stairs(counts / max(1, counts.sum()) * 100, h.edges, fill=True, color='#247c96')
            if row['q25'] is not None:
                ax.axvspan(row['q25'], row['q75'], color='#e69337', alpha=.35)
        ax.set_title(display_name(marker) + (' · inspect' if row['candidate'] else ' · ' + row['note']),
                     loc='left', fontweight='bold')
        if not row['n_finite']:
            ax.text(.5, .5, 'No measurements', ha='center', transform=ax.transAxes)
        ax.set_ylabel('% of measured cells')
        ax.set_xlabel('Analysis value · shading: middle 50%')
        other = bundle['summary'].query('marker == @marker')
        ax = axes[i, 1]
        for j, item in enumerate(other.itertuples()):
            ax.plot([item.q25, item.q75], [j, j], color='#247c96', lw=4)
            ax.plot(item.median, j, 'o', color='#243746', ms=4)
        ax.set_yticks(range(len(other)), other.phenotype, fontsize=8)
        if not other.n_finite.any():
            ax.text(.5, .5, 'No measurements', ha='center', transform=ax.transAxes)
        ax.set_xlabel('Median and middle 50% · selected phenotypes')
        ax = axes[i, 2]
        groups = bundle['patients'].query('phenotype == @phenotype and marker == @marker')
        groups = groups[groups.n_finite >= bundle['metadata']['min_cells']]
        if len(groups):
            ax.plot(range(len(groups)), groups['median'], 'o', color='#247c96', ms=4)
            ticks = np.unique(np.linspace(0, len(groups)-1, min(12, len(groups))).astype(int))
            ax.set_xticks(ticks, groups.patient.iloc[ticks], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Sample median')
            ax.set_xlabel(bundle['metadata']['patient_column'])
        else:
            ax.set_axis_off()
            reason = ('No samples meet the measured-cell minimum' if bundle['metadata']['patient_column']
                      else 'Choose an optional sample column')
            ax.text(.5, .5, 'Sample comparison not available\n' + reason,
                    ha='center', va='center', transform=ax.transAxes)
    fig.suptitle(f'{phenotype} · Marker review\nFirst {len(rows)} markers in inspection order · candidates, not positive calls', fontsize=16)
    fig.supxlabel('Before clustering normalization. Check raw staining and segmentation before choosing markers.\n'
                  'All selected markers, exact values and linked cell inspection are available in Studio.', fontsize=10)
    return fig
