"""Saved marker exploration, using the Sub-phenotyping cell and marker selections."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

from spacemap import dataio
from spacemap import marker_exploration as plots

FIGURES = {
    'review': 'Marker review',
    'dot': 'Dot plot',
    'histogram': 'Marker histograms',
    'patients': 'Patient heatmap',
    'groups': 'Group comparison',
    'composition': 'High-cell composition',
    'coexpression': 'Marker co-expression',
    'variation': 'Marker variation',
}
DEFAULTS = dict(patient_column='', z=1.5, min_cells=100, transform='auto', cofactor=5.,
                output_dir='', variation='iqr', pair=[], stratify=[], figures=['dot'])
TABLES = ('thresholds', 'summary', 'histograms', 'patients', 'coexpression', 'stratified', 'groups')


def options(body):
    if not isinstance(body, dict) or set(body) - set(DEFAULTS):
        raise ValueError('Choose valid marker figure settings.')
    found = {**DEFAULTS, **body}
    figures = found['figures']
    if (not isinstance(figures, list) or any(not isinstance(f, str) or f not in FIGURES for f in figures)
            or len(set(figures)) != len(figures)):
        raise ValueError('Choose figures from the list.')
    for name in ('patient_column', 'output_dir'):
        if not isinstance(found[name], str) or '\0' in found[name]:
            raise ValueError('Choose a patient column and output folder.')
    for name, low, high in [('z', -5, 10), ('min_cells', 1, 1000000), ('cofactor', .001, 65535)]:
        value = found[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f'{name.replace("_", " ")} must be between {low} and {high}.')
    if int(found['min_cells']) != found['min_cells']:
        raise ValueError('Minimum cells must be a whole number.')
    if found['transform'] not in ('auto', 'stored'):
        raise ValueError('Choose the expression scale.')
    if found['variation'] not in ('iqr', 'sd', 'relative_iqr', 'relative_sd'):
        raise ValueError('Choose IQR or standard deviation.')
    pair = found['pair']
    if (not isinstance(pair, list) or len(pair) not in (0, 2) or
            any(not isinstance(m, str) or not m for m in pair) or len(set(pair)) != len(pair)):
        raise ValueError('Choose two different markers for co-expression.')
    strata = found['stratify']
    if (not isinstance(strata, list) or len(strata) > 4 or len(set(strata)) != len(strata)
            or any(not isinstance(c, str) or not c for c in strata)
            or found['patient_column'] in strata):
        raise ValueError('Choose up to four group columns, separate from the patient column.')
    return found


def analysis_options(opts):
    """Ignore preferences for unselected views, while retaining them in saved settings."""
    effective = dict(opts)
    selected = set(opts.get('figures', FIGURES))  # Older queued jobs requested every view.
    if not selected & {'patients', 'groups'}:
        effective['stratify'] = []
    if not selected & {'patients', 'groups', 'histogram', 'review'}:
        effective['patient_column'] = ''
    if not selected & {'coexpression', 'review'}:
        effective['pair'] = []
    return effective


def prepare(root, table, config, image_settings=None):
    from .tasks import source_revision
    from .subphenotypes import check
    selected = {k: config[k] for k in ('parent_col', 'parent_values', 'markers', 'filters')}
    check(selected)
    opts = options({**config.get('figure_options', {}),
                    **{k: config[k] for k in ('transform', 'cofactor') if k in config}})
    if not opts['figures']:
        raise ValueError('Select at least one figure.')
    if 'patients' in opts['figures'] and not opts['patient_column']:
        raise ValueError('Choose a patient column for the patient heatmap.')
    if 'groups' in opts['figures'] and not opts['stratify']:
        raise ValueError('Choose a Stratify by column for group comparison.')
    if 'pair' not in config.get('figure_options', {}) and len(selected['markers']) >= 2:
        opts['pair'] = selected['markers'][:2]
    effective = analysis_options(opts)
    available = dataio.columns(table)
    required = [selected['parent_col'], *selected['markers'], *(f['column'] for f in selected['filters']), *effective['stratify']]
    if effective['patient_column']:
        required.append(effective['patient_column'])
    if set(required) - set(available):
        raise ValueError('Missing columns: ' + ', '.join(sorted(set(required) - set(available))))
    if not selected['parent_values'] or not selected['markers']:
        raise ValueError('Choose phenotypes and markers above first.')
    selected['markers'] = list(dict.fromkeys(selected['markers']))
    if set(selected['markers']) & {selected['parent_col'], effective['patient_column'], 'cell_id', 'batch_id', 'label'}:
        raise ValueError('Use measurements as markers, not label or ID columns.')
    if set(effective['stratify']) & {selected['parent_col'], *selected['markers']}:
        raise ValueError('Choose annotation columns for groups, separate from phenotypes and markers.')
    if 'coexpression' in opts['figures'] and len(effective['pair']) != 2:
        raise ValueError('Choose two different markers for co-expression.')
    if set(effective['pair']) - set(selected['markers']):
        raise ValueError('Choose co-expression markers from the selected marker panel.')
    out = Path(opts['output_dir']).expanduser() if opts['output_dir'].strip() else Path(root) / 'outputs/phenotyping/marker_figures'
    if not out.is_absolute():
        out = Path(root) / out
    if out.exists() and not out.is_dir():
        raise ValueError('Output must be a folder.')
    opts['output_dir'] = str(out.resolve())
    settings = dict(project_root=str(root), table=str(table), config=selected, options=opts,
                    markers=selected['markers'])
    settings.update(image_settings or {})
    settings['revision'] = source_revision(settings)
    return settings


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def recipes(bundle, variation, selected=None):
    meta = bundle['metadata']
    markers = meta['markers']
    selected = set(FIGURES if selected is None else selected)
    result = []

    def add(kind, phenotype=None, markers=markers, patients=None):
        title = plots.KINDS[kind] + (' · ' + phenotype if phenotype else '')
        if len(markers) < len(meta['markers']):
            title += ' · ' + plots.display_name(markers[0])
            if len(markers) > 1:
                title += '–' + plots.display_name(markers[-1])
        if patients and kind in ('patients','stratified'):
            title += ' · page ' + str(1 + len([e for e in result if e['kind']==kind and e['phenotype']==phenotype and e['markers']==markers]))
        suffix='_' + re.sub(r'[^\w.-]+', '_', phenotype).strip('_')[:60] if phenotype else ''
        result.append(dict(id=f'{len(result)+1:03d}_{kind}{suffix}', kind=kind, title=title, phenotype=phenotype,
                           markers=markers, patients=patients, variation=variation))
    if 'review' in selected:
        for phenotype in meta['phenotypes']:
            add('review', phenotype)
    for group in chunks(markers, 24):
        for kind in ('dot', 'composition'):
            if kind in selected:
                add(kind, markers=group)
    patient_kind = 'stratified' if meta['stratify'] else 'patients'
    patient_ids = list(meta['units']) if meta['stratify'] else sorted(bundle['patients'].patient.unique(), key=plots.natural_key)
    patient_groups = chunks(patient_ids, 16 if meta['stratify'] else 24) if meta['patient_column'] and 'patients' in selected else []
    strata_groups = chunks([s for s in meta['strata'] if s in set(bundle['groups'].stratum)], 16) if 'groups' in selected else []
    # Overview rows span the selected phenotypes, as in the patient recurrence figure.
    # Paginate saved plots so every selected marker is readable.
    overview_size = max(1, min(12, 24 // len(meta['phenotypes'])))
    for group in chunks(markers, overview_size):
        for pats in patient_groups:
            add(patient_kind, markers=group, patients=pats)
        for groups in strata_groups:
            add('groups', markers=group, patients=groups)
    for phenotype in meta['phenotypes']:
        if 'histogram' in selected:
            for group in chunks(markers, 12):
                add('histogram', phenotype, group)
        if 'variation' in selected:
            add('variation', phenotype)
        if 'coexpression' in selected and phenotype in set(bundle['coexpression'].phenotype):
            add('coexpression', phenotype, meta['pair'])
    return result


def run(settings, folder, report):
    from .tasks import source_revision
    from .subphenotypes import region_mask
    from .phenotyping_images import mask as image_mask
    from matplotlib.backends.backend_pdf import PdfPages

    if source_revision(settings) != settings['revision']:
        raise ValueError('The table changed. Generate the figures again.')
    config, opts = settings['config'], settings['options']
    effective = analysis_options(opts)
    selected = opts.get('figures', list(FIGURES))
    report('Reading selected cells…', .03)
    columns = list(dict.fromkeys([config['parent_col'], *config['markers'], *effective['stratify'],
        *(f['column'] for f in config['filters']), *([effective['patient_column']] if effective['patient_column'] else []),
        *([settings['roi_column']] if settings.get('roi_column') else [])]))
    frame = dataio.load(settings['table'], columns=columns)
    if source_revision(settings) != settings['revision']:
        raise ValueError('The table changed while loading. Generate the figures again.')
    mask = image_mask(frame, settings) & region_mask(frame, config['filters']) & frame[config['parent_col']].astype('string').isin(config['parent_values']).fillna(False)
    frame = frame.loc[mask].copy()
    bundle = plots.summarize(frame, config['parent_col'], config['markers'], patient=effective['patient_column'],
                            z=opts['z'], min_cells=opts['min_cells'], transform=opts['transform'],
                            cofactor=opts['cofactor'], pair=effective['pair'], stratify=effective['stratify'],
                            kinds=set(selected) | ({'histogram', 'coexpression'} if 'review' in selected else set()), report=report)
    del frame
    if not bundle['thresholds'].n_finite.sum():
        raise ValueError('These markers have no numeric measurements in the selected cells.')
    entries = recipes(bundle, opts['variation'], selected)
    if not entries:
        raise ValueError('No figures to draw. Check the selected patient/group columns and marker measurements.')
    now = datetime.now(timezone.utc)
    output = Path(opts['output_dir']) / ('marker_figures_' + now.strftime('%Y%m%d_%H%M%SZ_') + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    for name in TABLES:
        bundle[name].to_parquet(output / (name + '.parquet'), index=False)
    if 'review' in selected:
        from spacemap.marker_review import records, RULE
        (output / 'review.json').write_text(json.dumps({'version': 1, 'rule': RULE,
            'phenotypes': {p: records(bundle, p) for p in bundle['metadata']['phenotypes']}}, indent=2))
    manifest = dict(**bundle['metadata'], created=now.isoformat(), source=settings['table'],
                    source_revision=settings['revision'], selection=config, options=opts,
                    selected_images=settings.get('selected_images'), image_column=settings.get('roi_column'))
    notes = []
    missing = bundle['thresholds'].query('n_finite == 0').marker.tolist()
    if missing:
        notes.append('No measurements: ' + ', '.join(missing) + '.')
    if manifest['missing_patient_cells']:
        notes.append(f"{manifest['missing_patient_cells']:,} cells lack sample IDs.")
    if manifest['missing_stratum_cells']:
        notes.append(f"{manifest['missing_stratum_cells']:,} cells lack group values; omitted from grouped views.")
    produced = {'patients' if e['kind'] == 'stratified' else e['kind'] for e in entries}
    for kind in selected:
        if kind not in produced:
            notes.append(FIGURES[kind] + ': no usable data; skipped.')
    manifest['notes'] = notes
    (output / 'metadata.json').write_text(json.dumps(manifest, indent=2))
    with plots.plot_style(), PdfPages(output / 'figure_book.pdf') as book:
        for i, entry in enumerate(entries):
            report('Drawing ' + entry['title'] + '…', .65 + .33 * i / len(entries))
            fig = plots.draw(bundle, **{k: entry[k] for k in ('kind', 'phenotype', 'markers', 'patients', 'variation')})
            fig.savefig(output / (entry['id'] + '.png'), dpi=140, facecolor='white', bbox_inches='tight')
            fig.savefig(output / (entry['id'] + '.pdf'), facecolor='white', bbox_inches='tight')
            book.savefig(fig, facecolor='white', bbox_inches='tight')
            fig.clear()
    result = dict(table=settings['table'], output_dir=str(output), figures=entries, metadata=manifest)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2))
    report(f'{len(entries)} figures saved.', 1)
    return result


def saved_result(root, job_id):
    from . import jobs
    job = jobs.status(root, job_id)
    if job.get('kind') != 'marker_figures' or job.get('status') != 'done':
        raise ValueError('Choose a finished Marker figures job.')
    return job['result']


def render(root, job_id, body):
    """Render a display subset from immutable summaries, keeping the original cutoff."""
    import pandas as pd
    result = saved_result(root, job_id)
    output = Path(result['output_dir'])
    entry = next((e for e in result['figures'] if e['id'] == body.get('figure')), None)
    if not entry:
        raise ValueError('Choose a saved figure.')
    markers = body.get('markers', entry['markers'])
    if (not isinstance(markers, list) or not 1 <= len(markers) <= 200 or
            any(not isinstance(m, str) or m not in result['metadata']['markers'] for m in markers)):
        raise ValueError('Choose at least one saved marker.')
    markers = list(dict.fromkeys(markers))
    if entry['kind'] == 'coexpression':
        markers = entry['markers']
    spec = {k: entry[k] for k in ('kind', 'phenotype', 'patients', 'variation')}
    spec['markers'] = markers
    spec['appearance'] = plots.appearance_options(body.get('appearance', {}))
    if (entry['kind'] in ('patients', 'stratified', 'groups')
            and len(markers) * len(result['metadata']['phenotypes']) > 240):
        raise ValueError('Choose fewer markers for this overview (at most 240 phenotype–marker rows).')
    name = 'view_' + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    # Atomic files: another request never sees a partially written figure.
    with plots.plot_style():
        if not (output / (name + '.png')).is_file():
            bundle = {k: pd.read_parquet(output / (k + '.parquet')) for k in TABLES}
            bundle['metadata'] = result['metadata']
            fig = plots.draw(bundle, **spec)
            for ext in ('pdf', 'png'):
                path = output / (name + '.' + ext)
                temp = output / ('.' + name + '.tmp')
                fig.savefig(temp, format=ext, dpi=140, facecolor='white', bbox_inches='tight')
                temp.replace(path)
            fig.clear()
            (output / (name + '.json')).write_text(json.dumps({
                'figure': entry['id'], **spec, 'source_revision': result['metadata']['source_revision']}, indent=2))
    return dict(file=name, title=entry['title'], markers=markers, appearance=spec['appearance'])


def figure_file(root, job_id, name):
    """Only serve figure files belonging to a finished job in this project."""
    import re
    result = saved_result(root, job_id)
    allowed = {'figure_book.pdf'}
    allowed.update(e['id'] + '.' + ext for e in result['figures'] for ext in ('png', 'pdf'))
    if name not in allowed and not re.fullmatch(r'view_[0-9a-f]{20}\.(png|pdf)', name):
        raise ValueError('Choose a saved figure file.')
    output = Path(result['output_dir']).resolve()
    path = (output / name).resolve()
    if path.parent != output or not path.is_file():
        raise ValueError('This figure file is not available.')
    return path
