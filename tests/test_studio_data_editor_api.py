"""Linked tables keep identities intact while applying and saving ordinary edits."""
import pandas as pd
from starlette.testclient import TestClient
from spacemap_gui.viewer import api
from test_studio_commands import setup  # noqa: F401


def test_delete_reference_column_affects_every_image_but_keeps_other_results(setup):
    from spacemap import dataio, labels, pending_columns

    channel, table, _, _ = setup
    frame = pd.read_csv(table)
    frame['cell_id'] = range(len(frame))
    frame.to_csv(table, index=False)
    labels.append(table, [{'row':0, 'column':'phenotype_ref2', 'value':'A'},
                          {'row':4, 'column':'phenotype_ref2', 'value':'B'}])
    pending_columns.save(table, {'phenotype_keep':['Kept'] * len(frame)},
                         anchor_column='cell_id', anchor_values=frame.cell_id)
    route = f'/studio/{channel}/data/'
    one_image = {'source':{'combined':True}, 'rows':{'this_image':True,'column':'batch_id','value':'P11_roi_2'}}
    with TestClient(api.app) as client:
        before = client.post(route+'query', json=one_image).json()
        assert before['matched'] == 4
        result = client.post(route+'mutate', json={**one_image, 'operation':{'kind':'delete_columns', 'columns':['phenotype_ref2']}})
        assert result.status_code == 200, result.text
        whole = client.post(route+'query', json={**one_image, 'rows':{'this_image':False}}).json()
        assert whole['matched'] == 5 and 'phenotype_ref2' not in whole['columns']
        saved = client.post(route+'save', json={**one_image, 'mode':'overwrite', 'revision':whole['revision']})
        assert saved.status_code == 200, saved.text
    result = dataio.load(table)
    assert len(result) == 5 and 'phenotype_ref2' not in dataio.columns(table)
    assert result.phenotype_keep.eq('Kept').all()


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
