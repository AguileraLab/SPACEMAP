"""Figure jobs use registered project settings and expose only their own figure files."""
import json

from starlette.testclient import TestClient

from spacemap_gui.studio import jobs, subphenotypes
from spacemap_gui.studio.settings import revision
from spacemap_gui.viewer.api import app
from test_studio_commands import setup  # noqa: F401


def test_generation_requires_current_settings_and_existing_roles(setup, monkeypatch):
    channel, table, root, _ = setup
    config = subphenotypes.save(root, {
        'parent_col': 'phenotype_ml_consensus', 'parent_values': ['CD8'], 'markers': ['M0'],
        'figure_options': {'patient_column': 'batch_id', 'min_cells': 1}},
        revision(subphenotypes.values(root)))
    started = []
    monkeypatch.setattr(jobs, 'start', lambda root, kind, settings: started.append((kind, settings)) or {'id': 'test'})
    client = TestClient(app)
    url = f'/studio/{channel}/subpheno_figures'
    info = client.get(url).json()
    assert info['latest'] is None and info['defaults']['figures'] == ['dot']
    assert info['figure_choices']['histogram'] == 'Marker histograms'
    assert client.post(url, json={'revision': 'stale'}).status_code == 400
    answer = client.post(url, json={'revision': revision(config)})
    assert answer.status_code == 200, answer.text
    assert started[0][0] == 'marker_figures'
    settings = started[0][1]
    assert settings['table'] == str(table) and settings['config']['parent_values'] == ['CD8']
    assert settings['options']['pair'] == []  # one marker is sufficient
    assert settings['options']['figures'] == ['dot']
    config = subphenotypes.save(root, {'figure_options': {
        'stratify': ['missing_column'], 'figures': ['groups']}}, revision(config))
    assert client.post(url, json={'revision': revision(config)}).status_code == 400
    assert client.post('/studio/unregistered/subpheno_figures', json={}).status_code >= 400


def test_download_is_project_scoped_and_file_allowlisted(setup):
    channel, table, root, _ = setup
    job_id = 'a' * 32
    folder = jobs.root(root) / job_id
    folder.mkdir(parents=True)
    output = root / 'figures'
    output.mkdir()
    (output / '001_dot.png').write_bytes(b'\x89PNG\r\n\x1a\nexample')
    (folder / 'status.json').write_text(json.dumps({'id': job_id, 'kind': 'marker_figures', 'status': 'done',
        'result': {'table': str(table), 'output_dir': str(output), 'figures': [{'id': '001_dot'}]}}))
    client = TestClient(app)
    url = f'/studio/{channel}/subpheno_figure'
    response = client.get(url, params={'job_id': job_id, 'file': '001_dot.png'})
    assert response.status_code == 200 and response.headers['content-type'] == 'image/png'
    assert client.get(url, params={'job_id': job_id, 'file': '../cells.csv'}).status_code == 400
    assert client.get(url, params={'job_id': 'b' * 32, 'file': '001_dot.png'}).status_code == 400
    assert client.get('/studio/unregistered/subpheno_figure', params={'job_id': job_id, 'file': '001_dot.png'}).status_code >= 400
