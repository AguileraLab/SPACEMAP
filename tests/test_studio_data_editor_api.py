"""Linked tables keep identities intact while applying and saving ordinary edits."""
import pandas as pd
from starlette.testclient import TestClient
from spacemap_gui.viewer import api
from test_studio_commands import setup  # noqa: F401


def test_combined_table_edit_save_and_identity_guard(setup):
    channel, table, _, _ = setup
    base = {'source': {'combined': True}, 'rows': {'this_image': False}}
    route = f'/studio/{channel}/data/'
    with TestClient(api.app) as client:
        result = client.post(route + 'query', json=base).json()
        assert result['matched'] == 5
        for op in ({'kind': 'rename_column', 'column': 'label', 'name': 'broken'},
                   {'kind': 'delete_columns', 'columns': ['batch_id']},
                   {'kind': 'cells', 'changes': [{'row': 0, 'col': 'label', 'value': 8}]}):
            response = client.post(route + 'mutate', json={**base, 'operation': op})
            assert response.status_code == 400 and 'links cells' in response.json()['error']
        response = client.post(route + 'mutate', json={**base, 'operation': {
            'kind': 'replace', 'column': 'phenotype_ml_consensus', 'old': 'CD8', 'value': 'CD8 T', 'scope': 'all'}})
        assert response.status_code == 200 and response.json()['affected'] == 2
        changed = client.post(route + 'query', json=base).json()
        assert changed['identity']['0']['label'] == 1
        response = client.post(route + 'save', json={**base, 'mode': 'overwrite', 'revision': changed['revision']})
        assert response.status_code == 200, response.text
        assert pd.read_csv(table).phenotype_ml_consensus.tolist()[:2] == ['CD8 T', 'CD8 T']
        assert list((table.parent / 'previous').rglob(table.name))
        export = client.post(route + 'export', json=base)
        assert export.status_code == 200 and 'CD8 T' in export.text
