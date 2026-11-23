"""Publish completed spatial runs under the chosen Output/method directory.

Job records remain in their existing location for progress, review and recovery.
This adds readable exports without moving or merging any source cell table.
"""
import json
from pathlib import Path
import shutil

from .spatial_sources import OUTPUT_KINDS, output_folder


def publish(project, kind, settings, job_folder, result):
    if kind not in OUTPUT_KINDS:
        return result
    base = Path(settings.get('spatial_output') or output_folder(project))
    existing = result.get('folder')
    # Density/triads/screening already publish full runs in their own subfolder.
    if existing and Path(existing).resolve().is_relative_to(base.resolve() / OUTPUT_KINDS[kind]):
        return {**result, 'export_folder': existing}
    source = Path(job_folder)
    target = base / OUTPUT_KINDS[kind] / ('studio_' + source.name)
    target.mkdir(parents=True, exist_ok=True)
    names = {'tls_candidates': ['tls_candidates.csv', 'tls_candidates.geojson', 'tls_method.json'],
             'cell_interactions': ['cci_all_pairs.csv']}.get(kind, [])
    if kind == 'gland_types':
        names = [p.name for p in source.iterdir() if p.is_file() and
                 (p.name.startswith(('gland_', 'interior_', 'exterior_', 'clinical')) or p.name == 'sample_burdens.csv')]
    for name in names:
        if (source / name).is_file():
            shutil.copyfile(source / name, target / name)
    (target / 'settings.json').write_text(json.dumps(settings, indent=2))
    (target / 'result.json').write_text(json.dumps(result, indent=2))
    if kind == 'neighbourhoods' and result.get('profile'):
        import pandas as pd
        profile = result['profile']
        pd.DataFrame(profile['raw'], index=profile['rows'], columns=profile['columns']).to_csv(target / 'neighbourhood_profiles.csv')
    return {**result, 'export_folder': str(target)}
