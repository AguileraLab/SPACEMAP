"""The data viewer: pages of a cell table, this image's rows, filters, sorting, a cell's row, edits, save, export."""

from pathlib import Path

import pandas as pd

from spacemap_gui.studio import data_view


def _table(tmp_path, suffix=".csv"):
    frame = pd.DataFrame({"batch_id": ["A"] * 6 + ["B"] * 4, "label": list(range(1, 7)) + list(range(1, 5)),
                          "kind": list("TTSSTSTTSS"), "CD3": [5.0, 50, 7, 70, 9, 90, 1, 2, 3, 4]})
    path = tmp_path / ("cells" + suffix)
    frame.to_csv(path, index=False) if suffix == ".csv" else frame.to_parquet(path, index=False)
    return path


def test_pages_filters_sort_and_find(tmp_path):
    path = _table(tmp_path, ".parquet")
    page = data_view.query(path, "batch_id", "A", [{"col": "CD3", "lo": 8, "hi": None}], {"col": "CD3", "desc": True}, 0, 2)
    assert page["matched"] == 4 and page["total"] == 6 and [r[page["shown"].index("CD3") + 1] for r in page["rows"]] == [90.0, 70.0]
    assert data_view.query(path, "batch_id", "A", [{"col": "kind", "contains": "t"}])["matched"] == 3
    assert data_view.query(path, "batch_id", "A", [{"col": "kind", "values": ["S"]}])["matched"] == 3
    assert data_view.find(path, "batch_id", "A", [], {"col": "CD3", "desc": True}, 6) == 0
    assert sorted(data_view.labels_of(path, "batch_id", "A", [{"col": "CD3", "lo": 60}])) == [4, 6]


def test_edits_are_saved_back_with_the_original_kept_or_as_a_copy(tmp_path):
    path = _table(tmp_path)
    row = data_view.query(path, "batch_id", "B")["rows"][0][0]
    assert data_view.edit(path, [{"row": row, "col": "kind", "value": "Tumor"}]) == 1
    assert data_view.query(path, "batch_id", "B", [{"col": "kind", "values": ["Tumor"]}])["matched"] == 1
    copy = data_view.save(path, "copy")
    assert Path(copy).name == "cells_edited.csv" and pd.read_csv(copy)["kind"].tolist().count("Tumor") == 1
    assert pd.read_csv(path)["kind"].tolist().count("Tumor") == 0  # the file itself is unchanged by a copy
    data_view.edit(path, [{"row": row, "col": "CD3", "value": "99"}])
    data_view.save(path, "overwrite")
    assert pd.read_csv(path).loc[row, "CD3"] == 99 and any((tmp_path / "previous").rglob("cells.csv"))
    text = data_view.export(path, "batch_id", "A", [{"col": "CD3", "lo": 60}], None, ["label", "CD3"])
    assert text.splitlines()[0] == "label,CD3" and len(text.splitlines()) == 3


def _large(tmp_path):
    """A Parquet table in many row groups (two images interleaved), as a merged table is."""
    import numpy as np

    count = 50_000
    frame = pd.DataFrame({"batch_id": np.where(np.arange(count) % 2, "B", "A"), "label": np.arange(count) // 2 + 1,
                          "kind": np.array(["T", "S", "Endo", ""])[np.arange(count) % 4], "CD3": (np.arange(count) % 997).astype(float)})
    path = tmp_path / "big.parquet"
    frame.to_parquet(path, index=False, row_group_size=4_000)
    return path, frame


def test_large_table_reads_only_what_a_page_needs(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    path, frame = _large(tmp_path)
    read = []
    original = pq.ParquetFile.read_row_group
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", lambda self, i, columns=None, **k: read.append((i, tuple(columns or ()))) or original(self, i, columns=columns, **k))
    page = data_view.query(path, "batch_id", "B", [{"col": "CD3", "lo": 990}], {"col": "CD3", "desc": True}, 0, 5, ["label", "CD3"])
    expected = frame[(frame.batch_id == "B") & (frame.CD3 >= 990)].sort_values("CD3", ascending=False, kind="stable")
    assert page["matched"] == len(expected) and page["total"] == 25_000
    assert [r[0] for r in page["rows"]] == expected.index[:5].tolist() and [r[2] for r in page["rows"]] == expected.CD3[:5].tolist()
    assert all(columns == ("label",) for _, columns in read) and len(read) <= 5  # CD3 was read for the filter; label for 5 rows only
    info = data_view.info(path)
    assert info["image_column"] == "batch_id" and info["values"] == ["A", "B"] and info["rows"] == 50_000


def test_summary_text_sort_and_edits_on_a_large_table(tmp_path):
    path, frame = _large(tmp_path)
    numbers = data_view.summary(path, "batch_id", "A", [], "CD3")
    mine = frame[frame.batch_id == "A"].CD3
    assert numbers["min"] == mine.min() and numbers["max"] == mine.max() and abs(numbers["mean"] - mine.mean()) < 1e-9
    groups = data_view.summary(path, "batch_id", "A", [], "kind")
    assert dict(groups["groups"]) == {"T": 12_500, "Endo": 12_500} and groups["empty"] == 0
    page = data_view.query(path, "batch_id", "B", [], {"col": "kind", "desc": False}, 0, 3, ["kind"])
    assert [r[1] for r in page["rows"]] == ["", "", ""] or [r[1] for r in page["rows"]] == ["S", "S", "S"]
    row = page["rows"][0][0]
    data_view.edit(path, [{"row": row, "col": "kind", "value": "Tumor"}, {"row": row, "col": "CD3", "value": "5000"}])
    assert data_view.query(path, "batch_id", "B", [{"col": "kind", "values": ["Tumor"]}])["matched"] == 1
    assert data_view.summary(path, "batch_id", "B", [], "CD3")["max"] == 5000
    copy = pd.read_parquet(data_view.save(path, "copy"))
    assert copy.loc[row, "kind"] == "Tumor" and copy.loc[row, "CD3"] == 5000 and len(copy) == 50_000
    data_view.edit(path, [{"row": 0, "col": "CD3", "value": "high"}])  # a word in a number column: the column becomes text
    assert pd.read_parquet(data_view.save(path, "copy")).loc[0, "CD3"] == "high"


def test_a_large_csv_opens_once_its_fast_copy_is_ready(tmp_path, monkeypatch):
    import time

    from spacemap import dataio

    monkeypatch.setattr(dataio, "FAST_COPY_MIN_BYTES", 10)
    path = _table(tmp_path)
    try:
        data_view.query(path, "batch_id", "A")
        ready = True
    except data_view.Preparing:
        ready = False
    for _ in range(200):
        if ready:
            break
        time.sleep(0.05)
        try:
            data_view.query(path, "batch_id", "A")
            ready = True
        except data_view.Preparing:
            pass
    assert ready and data_view.query(path, "batch_id", "A")["total"] == 6


def test_a_filter_that_keeps_no_rows_gives_an_empty_page(tmp_path):
    path = _table(tmp_path, ".parquet")
    page = data_view.query(path, "batch_id", "A", [{"col": "kind", "values": ["none"]}])
    assert page["matched"] == 0 and page["rows"] == []


def test_saved_phenotypes_and_scores_are_paged_filtered_exported_and_refreshed(tmp_path):
    from spacemap import labels, pending_columns
    path = _table(tmp_path, ".parquet")
    frame = pd.read_parquet(path)
    frame["cell_id"] = [f"c{i}" for i in range(len(frame))]
    frame.to_parquet(path, index=False, row_group_size=3)
    assert "phenotype_test" not in data_view.info(path)["columns"]
    pending_columns.save(path, {"phenotype_test": ["T"] * 6 + ["B"] * 4, "prediction_score": [0.9] * 10},
                         anchor_column="cell_id", anchor_values=frame.cell_id)
    page = data_view.query(path, "batch_id", "B", columns=["phenotype_test", "prediction_score"])
    assert page["rows"] == [[i, "B", 0.9] for i in range(6, 10)]
    assert page["kinds"]["prediction_score"] == "value"
    labels.append(path, [{"row": 6, "column": "phenotype_test", "value": "Edited"}])
    selected = data_view.query(path, "batch_id", "B", [{"col": "phenotype_test", "values": ["Edited"]}])
    assert selected["matched"] == 1 and selected["rows"][0][0] == 6
    assert "Edited" in data_view.export(path, "batch_id", "B", columns=["phenotype_test"])
    copy = pd.read_parquet(data_view.save(path, "copy"))
    assert copy.phenotype_test.tolist() == ["T"] * 6 + ["Edited"] + ["B"] * 3
    assert copy.prediction_score.eq(0.9).all()
    assert "phenotype_test" not in pd.read_parquet(path).columns
    pending_columns.save(path, {"prediction_score": [0.2] * 10}, anchor_column="cell_id", anchor_values=frame.cell_id)
    assert data_view.summary(path, "batch_id", "A", [], "prediction_score")["max"] == 0.2


def test_custom_local_label_is_scoped_by_roi(tmp_path):
    path = _table(tmp_path, ".parquet")
    frame = pd.read_parquet(path).rename(columns={"label": "object_number", "batch_id": "roi_name"})
    frame.to_parquet(path, index=False)
    assert data_view.find(path, "roi_name", "B", [], None, 1, "object_number") == 0
    assert data_view.labels_of(path, "roi_name", "B", [{"col":"kind", "values":["T"]}], "object_number") == [1, 2]


def test_bulk_replace_scopes_schema_edits_row_delete_and_history(tmp_path):
    path=_table(tmp_path,'.parquet')
    first=data_view.query(path)
    result=data_view.mutate(path,{'kind':'replace','column':'kind','old':'T','value':'CD8','scope':'filtered'},'batch_id','A',[{'col':'CD3','lo':8}],revision=first['revision'])
    assert result['affected']==2
    assert data_view.summary(path,column='kind')['groups'][0][0]=='S'
    assert data_view.query(path,filters=[{'col':'kind','values':['CD8']}])['matched']==2
    data_view.history(path,'undo')
    assert data_view.query(path,filters=[{'col':'kind','values':['CD8']}])['matched']==0
    data_view.history(path,'redo')
    assert data_view.query(path,filters=[{'col':'kind','values':['CD8']}])['matched']==2
    result=data_view.mutate(path,{'kind':'replace','column':'kind','old':'T','value':'T cell','scope':'all'},'batch_id','B')
    assert result['affected']==3
    data_view.mutate(path,{'kind':'rename_column','column':'kind','name':'phenotype_ref'})
    assert 'phenotype_ref' in data_view.query(path)['columns'] and 'kind' not in data_view.query(path)['columns']
    data_view.mutate(path,{'kind':'delete_columns','columns':['CD3']})
    assert 'CD3' not in data_view.query(path)['columns']
    data_view.history(path,'undo');assert 'CD3' in data_view.query(path)['columns']
    result=data_view.mutate(path,{'kind':'delete_rows','scope':'selected'},selected=[1,6,7])
    assert result['affected']==3 and data_view.query(path)['matched']==7
    assert [r[0] for r in data_view.query(path)['rows']]==[0,2,3,4,5,8,9]
    assert len(pd.read_parquet(path))==10
    data_view.history(path,'undo');assert data_view.query(path)['matched']==10
    data_view.history(path,'redo')
    copy=Path(data_view.save(path,'copy',name='cleaned.parquet'))
    frame=pd.read_parquet(copy)
    assert len(frame)==7 and 'phenotype_ref' in frame and 'kind' not in frame
    assert pd.read_parquet(path).columns.tolist()==['batch_id','label','kind','CD3']


def test_save_materializes_overlays_before_row_deletion_and_keeps_backup(tmp_path):
    from spacemap import labels,pending_columns
    path=_table(tmp_path,'.parquet');frame=pd.read_parquet(path)
    frame['cell_id']=[f'c{i}' for i in range(len(frame))];frame.to_parquet(path,index=False)
    pending_columns.save(path,{'phenotype_ml':['A']*6+['B']*4},anchor_column='cell_id',anchor_values=frame.cell_id)
    labels.append(path,[{'row':7,'column':'phenotype_ml','value':'Plasma'}])
    data_view.mutate(path,{'kind':'replace','column':'phenotype_ml','old':'B','value':'B cell','scope':'all'})
    data_view.mutate(path,{'kind':'delete_rows','scope':'selected'},selected=[0,6])
    data_view.save(path,'overwrite')
    saved=pd.read_parquet(path)
    assert len(saved)==8 and saved.phenotype_ml.tolist()==['A']*5+['Plasma','B cell','B cell']
    assert not labels.load(path).shape[0] and not pending_columns.pending_names(path)
    backups=list((tmp_path/'previous').rglob(path.name));assert len(backups)==1
    assert pd.read_parquet(backups[0]).equals(frame)
    assert list(backups[0].parent.glob('*.pending'))


def test_edits_are_atomic_and_conflicts_are_rejected(tmp_path):
    import pytest
    path=_table(tmp_path,'.parquet')
    initial=data_view.query(path)
    with pytest.raises(ValueError,match='row'):
        data_view.mutate(path,{'kind':'cells','changes':[{'row':0,'col':'kind','value':'bad'},{'row':999,'col':'kind','value':'bad'}]})
    assert data_view.query(path)['rows'][0][3]=='T'
    data_view.mutate(path,{'kind':'fill','column':'kind','value':'Reviewed','scope':'selected'},selected=[2])
    with pytest.raises(ValueError,match='changed'):
        data_view.mutate(path,{'kind':'delete_rows','scope':'all'},revision=initial['revision'])
    frame=pd.read_parquet(path);frame.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='outside Data'):data_view.query(path)
    data_view.history(path,'discard');assert data_view.query(path)['matched']==10


def test_new_edit_after_undo_discards_redo_and_integer_ids_are_exact(tmp_path):
    path=tmp_path/'large_ids.parquet';pd.DataFrame({'id':[2**53+1,2**53+2],'value':['one','two']}).to_parquet(path,index=False)
    assert data_view.query(path)['rows'][0][1]==2**53+1
    data_view.edit(path,[{'row':0,'col':'id','value':str(2**53+3)}]);data_view.history(path,'undo')
    assert data_view.query(path)['rows'][0][1]==2**53+1
    data_view.edit(path,[{'row':1,'col':'value','value':'second'}])
    assert not data_view.query(path)['can_redo']
    data_view.history(path,'undo');assert not data_view.query(path)['edited']


def test_disk_backed_bulk_edit_does_not_read_whole_columns(tmp_path,monkeypatch):
    import pyarrow.parquet as pq
    from spacemap_gui.studio import data_editor
    path,frame=_large(tmp_path)
    monkeypatch.setattr(pq.ParquetFile,'read',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Whole-column read')))
    monkeypatch.setattr(pq.ParquetFile,'read_row_group',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Unbounded row group')))
    result=data_view.mutate(path,{'kind':'replace','column':'kind','old':'T','value':'CD8','scope':'all'})
    assert result['affected']==12500
    page=data_view.query(path,filters=[{'col':'kind','values':['CD8']}],sort={'col':'CD3','desc':True},limit=20)
    assert page['matched']==12500 and len(page['rows'])==20
    e=data_editor.editor(path)
    assert e.db.execute('PRAGMA temp_store').fetchone()[0]==1
    assert Path(e.directory.name,'workspace.sqlite').stat().st_size>0
    copy=data_view.save(path,'copy');assert len(pd.read_parquet(copy))==50000


def test_save_failure_keeps_original_and_undo_then_empty_save_keeps_schema(tmp_path, monkeypatch):
    import pytest
    from spacemap_gui.studio import data_editor
    path = _table(tmp_path, '.parquet')
    original = path.read_bytes()
    data_view.mutate(path, {'kind': 'delete_rows', 'scope': 'all'})
    with monkeypatch.context() as patch:
        patch.setattr(data_editor.os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('Disk failure')))
        with pytest.raises(OSError, match='Disk failure'):
            data_view.save(path, 'overwrite')
    assert path.read_bytes() == original and data_view.query(path)['matched'] == 0
    data_view.history(path, 'undo')
    assert data_view.query(path)['matched'] == 10
    data_view.history(path, 'redo')
    previous = data_editor.editor(path)
    data_view.save(path, 'overwrite')
    assert pd.read_parquet(path).empty and list(pd.read_parquet(path)) == ['batch_id', 'label', 'kind', 'CD3']
    with pytest.raises(ValueError, match='saved or reloaded'):
        previous.check()


def test_save_pending_columns_with_different_row_groups(tmp_path):
    from spacemap import pending_columns
    path = tmp_path / 'boundaries.parquet'
    frame = pd.DataFrame({'cell_id': [f'c{i}' for i in range(35001)], 'value': range(35001)})
    frame.to_parquet(path, index=False, row_group_size=17003)
    values = [str(i) for i in range(len(frame))]
    pending_columns.save(path, {'phenotype': values}, anchor_column='cell_id', anchor_values=frame.cell_id)
    data_view.mutate(path, {'kind': 'delete_rows', 'scope': 'selected'}, selected=[16383, 17002])
    copy = data_view.save(path, 'copy')
    saved = pd.read_parquet(copy)
    assert saved.phenotype.tolist() == [v for i, v in enumerate(values) if i not in (16383, 17002)]
