"""Saved predictions refresh independently from drafts; actual conflicts survive."""
import pandas as pd
import pytest

from spacemap import labels, pending_columns
from spacemap_gui.studio import data_editor, data_view


@pytest.fixture
def table(tmp_path):
    path = tmp_path / 'cells.parquet'
    pd.DataFrame({'batch_id':['B','A','B','A'], 'cell_id':['a','b','c','d'],
                  'label':[1,1,2,2], 'manual':['T','B','T','B'], 'marker':[1.,2.,3.,4.]}).to_parquet(path, row_group_size=2)
    return path


def publish(table, **columns):
    pending_columns.save(table, columns, anchor_column='cell_id', anchor_values=['a','b','c','d'])


def test_pending_prediction_preserves_draft_undo_redo_and_saved_values(table):
    data_view.edit(table,[{'row':1,'col':'manual','value':'Reviewed'}])
    editor = data_editor.editor(table)
    publish(table, phenotype_ml=['P','Q','P','Q'])
    page = data_view.query(table,'batch_id','A',columns=['manual','phenotype_ml'])
    assert page['rows'] == [[1,'Reviewed','Q'],[3,'B','Q']]
    assert data_editor.editor(table) is editor and page['edited'] == 1
    data_view.history(table,'undo')
    assert data_view.query(table,columns=['manual','phenotype_ml'])['rows'][1] == [1,'B','Q']
    data_view.history(table,'redo')
    saved = pd.read_parquet(data_view.save(table,'copy'))
    assert saved.manual.tolist() == ['T','Reviewed','T','B']
    assert saved.phenotype_ml.tolist() == ['P','Q','P','Q']
    assert pd.read_parquet(table).manual.tolist() == ['T','B','T','B']


def test_replaced_unedited_result_invalidates_loaded_values_but_keeps_draft(table):
    publish(table, phenotype_ml=['P']*4)
    data_view.summary(table,column='phenotype_ml')
    data_view.edit(table,[{'row':0,'col':'manual','value':'Reviewed'}])
    publish(table, phenotype_ml=['Q']*4)
    assert data_view.summary(table,column='phenotype_ml')['groups'] == [('Q',4)]
    assert data_view.query(table,columns=['manual'])['rows'][0] == [0,'Reviewed']


@pytest.mark.parametrize('change',['prediction','labels','base','rename_collision','redo'])
def test_actual_conflicts_never_discard_or_rebase_drafts(table,change):
    data_view.edit(table,[{'row':0,'col':'manual','value':'Reviewed'}])
    editor=data_editor.editor(table)
    if change=='prediction': publish(table,manual=['Changed']*4)
    elif change=='labels': labels.append(table,[{'row':0,'column':'manual','value':'Changed'}])
    elif change=='base':
        frame=pd.read_parquet(table).iloc[::-1];frame.to_parquet(table)
    elif change=='rename_collision':
        data_view.mutate(table,{'kind':'rename_column','column':'manual','name':'new_prediction'})
        publish(table,new_prediction=['Changed']*4)
    else:
        data_view.edit(table,[{'row':0,'col':'marker','value':5}])
        data_view.history(table,'undo')
        publish(table,marker=[9]*4)
    with pytest.raises(data_editor.ExternalChange):data_view.query(table)
    assert data_editor.SESSIONS[str(table.resolve())] is editor and not editor.closed
    assert editor.cursor > 0
    data_view.history(table,'discard')
    assert data_view.query(table)['edited']==0


def test_saved_image_names_ignore_conflicting_data_drafts(table,monkeypatch):
    data_view.edit(table,[{'row':0,'col':'manual','value':'Reviewed'}])
    publish(table,manual=['Changed']*4)
    monkeypatch.setattr(data_editor,'editor',lambda *a,**k:pytest.fail('Saved image lookup must not open Data'))
    assert data_view.saved_image_values(table,'batch_id') == ['A','B']


def test_image_info_and_page_do_not_build_a_full_sqlite_column(table,monkeypatch):
    monkeypatch.setattr(data_editor.Editor,'ensure',lambda *a,**k:pytest.fail('Simple image opening must not materialize a column'))
    assert data_view.info(table)['values'] == ['A','B']
    page=data_view.query(table,'batch_id','A',columns=['label','manual'])
    assert page['rows']==[[1,1,'B'],[3,2,'B']]


def test_no_image_value_keeps_the_requested_filter(table):
    page=data_view.query(table,'batch_id',None,[{'col':'manual','values':['T']}],columns=['manual'])
    assert page['rows']==[[0,'T'],[2,'T']]


def test_new_predictions_keep_renamed_and_deleted_column_history(table):
    data_view.mutate(table,{'kind':'rename_column','column':'manual','name':'reviewed'})
    data_view.mutate(table,{'kind':'delete_columns','columns':['marker']})
    publish(table,phenotype_ml=['P']*4)
    assert data_view.query(table)['columns'] == ['batch_id','cell_id','label','reviewed','phenotype_ml']
    data_view.history(table,'undo');data_view.history(table,'undo')
    assert data_view.query(table)['columns'] == ['batch_id','cell_id','label','manual','marker','phenotype_ml']


def test_result_arriving_during_open_is_included_in_consistent_snapshot(table,monkeypatch):
    source=data_editor.Source
    calls=[]
    def opening(path):
        result=source(path)
        calls.append(path)
        if len(calls)==1:publish(table,phenotype_new=['T']*4)
        return result
    monkeypatch.setattr(data_editor,'Source',opening)
    page=data_view.query(table,columns=['phenotype_new'])
    assert len(calls)==2
    assert page['rows']==[[i,'T'] for i in range(4)]
