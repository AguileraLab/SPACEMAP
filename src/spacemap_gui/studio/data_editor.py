"""Paged, disk-backed table editing. Raw files change only on explicit Save.

Arrow reads bounded batches. SQLite stores only columns needed for queries or
edits, plus disk-backed selections and undo records; it spills sorts to disk.
Row identifiers remain original file positions until a save finishes.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BATCH = 16384
SESSIONS = OrderedDict()
LOCK = threading.RLock()


def stamp(path):
    from spacemap.viewer.cells import _edits_token
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns, *_edits_token(str(path)))


def kind(dtype):
    return 'value' if pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype) else 'group'


def plain(dtype):
    return dtype.value_type if pa.types.is_dictionary(dtype) else dtype


def scalar(value, dtype):
    """Validate a scalar without going through float for integer identifiers."""
    if value is None or value == '' and kind(dtype) == 'value':
        return None, dtype
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_null(dtype):
        return str(value), pa.string()
    if pa.types.is_boolean(dtype):
        text = str(value).lower()
        if text not in ('true', 'false', '1', '0', 'yes', 'no'):
            raise ValueError('Enter true or false for a boolean column.')
        return text in ('true', '1', 'yes'), dtype
    if pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
        from decimal import Decimal, InvalidOperation
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return str(value), pa.string()
        if not number.is_finite():
            raise ValueError('Enter a finite number, or clear the value.')
        if pa.types.is_integer(dtype) and number == number.to_integral_value():
            converted = int(number)
            # Preserve integer precision, promote small integer types when needed.
            try:
                pa.scalar(converted, type=dtype)
            except (pa.ArrowInvalid, OverflowError):
                dtype = pa.int64()
                pa.scalar(converted, type=dtype)
            return converted, dtype
        return float(number), pa.float64()
    result = pa.array([value]).cast(dtype)[0].as_py()
    return result.isoformat() if hasattr(result, 'isoformat') else result, dtype


def arrow(values, dtype):
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return pa.array([None if v is None else str(v) for v in values], type=dtype)
    if pa.types.is_boolean(dtype):
        return pa.array([None if v is None else bool(v) for v in values], type=dtype)
    if pa.types.is_integer(dtype):
        return pa.array([None if v is None else int(v) for v in values], type=dtype)
    if pa.types.is_floating(dtype):
        return pa.array([None if v is None else float(v) for v in values], type=dtype, from_pandas=True)
    return pa.array(values).cast(dtype)


class Source:
    def __init__(self, path):
        from .data_view import _Table
        from spacemap import pending_columns, labels
        from spacemap import dataio
        if Path(path).suffix.lower()!='.parquet' and Path(path).stat().st_size>dataio.FAST_COPY_MIN_BYTES and not dataio.fast_copies_enabled():
            raise ValueError('Open a Parquet copy of this large CSV, or enable fast table copies before editing it.')
        self.base = _Table(path, '', None)  # metadata only for Parquet
        self.path = str(path)
        self.length = self.base.length
        self.pending = {p['column']: pq.ParquetFile(p['file']) for p in pending_columns.pending(path) if not p['stale']}
        self.fields = {}
        for name in self.base.names:
            dtype = (self.pending[name].schema_arrow.field('value').type if name in self.pending else
                     self.base.schema.field(name).type if name in self.base.schema.names else pa.string())
            self.fields[name] = plain(dtype)
        sidecar = labels.sidecar_path(path)
        self.labels = None
        if sidecar.is_file():
            file = pq.ParquetFile(sidecar)
            recorded = (file.schema_arrow.metadata or {}).get(labels._FINGERPRINT_KEY)
            if recorded is None or recorded.decode() == labels._fingerprint(Path(path)):
                self.labels = file

    def batches(self, names):
        physical = [n for n in names if n in self.base.schema.names and n not in self.pending]
        if self.base.file is not None and physical:
            chunks = self.base.file.iter_batches(batch_size=BATCH, columns=physical, use_threads=False)
        elif self.base.memory is not None and physical:
            chunks = iter(self.base.memory.select(physical).to_batches(max_chunksize=BATCH))
        else:
            chunks = (None for _ in range(0, self.length, BATCH))
        overlays = {n: self.pending[n].iter_batches(batch_size=BATCH, columns=['value'], use_threads=False) for n in names if n in self.pending}
        remainders={}
        start = 0
        for chunk in chunks:
            size = chunk.num_rows if chunk is not None else min(BATCH, self.length - start)
            arrays = {}
            for name in names:
                if name in overlays:
                    needed=size;pieces=[]
                    while needed:
                        part=remainders.pop(name,None)
                        if part is None:
                            batch=next(overlays[name],None)
                            if batch is None:raise ValueError('A saved result column is shorter than the table.')
                            part=batch.column(0)
                        take=min(needed,len(part));pieces.append(part.slice(0,take));needed-=take
                        if take<len(part):remainders[name]=part.slice(take)
                    arrays[name]=pa.concat_arrays(pieces).cast(self.fields[name])
                elif name in physical:
                    arrays[name] = chunk.column(chunk.schema.get_field_index(name)).cast(self.fields[name])
                else:
                    arrays[name] = pa.nulls(size, self.fields[name])
            yield start, size, arrays
            start += size

    @staticmethod
    def take_file(file, ids, names):
        sizes = [file.metadata.row_group(i).num_rows for i in range(file.num_row_groups)]
        starts = np.r_[0, np.cumsum(sizes)]
        groups = np.searchsorted(starts, ids, side='right') - 1
        output = {n: [None] * len(ids) for n in names}
        for group in np.unique(groups):
            start = int(starts[group])
            needed = np.flatnonzero(groups == group)
            for batch in file.iter_batches(batch_size=BATCH, row_groups=[int(group)], columns=names, use_threads=False):
                inside = needed[(ids[needed] >= start) & (ids[needed] < start + batch.num_rows)]
                if len(inside):
                    positions = pa.array(ids[inside] - start)
                    for n in names:
                        vals = batch.column(batch.schema.get_field_index(n)).take(positions).to_pylist()
                        for i, v in zip(inside, vals):
                            output[n][int(i)] = v
                start += batch.num_rows
                if start > int(ids[needed].max()):
                    break
        return output

    def take(self, ids, names):
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return {n: [] for n in names}
        physical = [n for n in names if n in self.base.schema.names and n not in self.pending]
        result = ({n: self.base.memory.column(n).take(pa.array(ids)).to_pylist() for n in physical}
                  if self.base.memory is not None else self.take_file(self.base.file, ids, physical) if physical else {})
        for name in names:
            if name in self.pending:
                result[name] = self.take_file(self.pending[name], ids, ['value'])['value']
            elif name not in result:
                result[name] = [None] * len(ids)
        return result


class Editor:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.stamp = stamp(self.path)
        self.source = Source(self.path)
        self.lock = threading.RLock()
        self.directory = tempfile.TemporaryDirectory(prefix='spacemap-data-')
        self.db = sqlite3.connect(str(Path(self.directory.name) / 'workspace.sqlite'), check_same_thread=False)
        self.db.execute('PRAGMA cache_size=-32768')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('PRAGMA mmap_size=0')
        self.db.execute('CREATE TABLE deleted(r INTEGER PRIMARY KEY)')
        self.db.execute('CREATE TABLE changes(h INTEGER,c INTEGER,r INTEGER,old,new,PRIMARY KEY(h,c,r))')
        self.db.execute('CREATE TABLE label_edits(c TEXT,r INTEGER,v,ts TEXT,PRIMARY KEY(c,r))')
        if self.source.labels:
            for batch in self.source.labels.iter_batches(batch_size=BATCH, use_threads=False):
                data = batch.to_pydict()
                self.db.executemany('INSERT INTO label_edits VALUES(?,?,?,?) ON CONFLICT(c,r) DO UPDATE SET v=excluded.v,ts=excluded.ts WHERE excluded.ts>=label_edits.ts',
                                    ((c, int(r), v, str(ts)) for c, r, v, ts in zip(data['column'],data['row'],data['value'],data['saved_at'])))
        self.db.commit()
        self.label_columns={r[0] for r in self.db.execute('SELECT DISTINCT c FROM label_edits')}
        self.columns = [{'name': n, 'original': n, 'id': i, 'type': t} for i, (n, t) in enumerate(self.source.fields.items())]
        self.loaded = set()
        self.history = []
        self.cursor = 0
        self.revision = 0
        self.views = OrderedDict()
        self.serial = 0
        self.progress = None
        self.closed = False

    def check(self):
        if self.closed:
            raise ValueError('This Data workspace was saved or reloaded. Refresh the table before continuing.')
        if stamp(self.path) != self.stamp:
            raise ValueError('This table or its saved labels changed outside Data. Discard these edits and reopen before continuing.')

    @property
    def names(self):
        return [c['name'] for c in self.columns]

    def column(self, name):
        found = next((c for c in self.columns if c['name'] == name), None)
        if found is None:
            raise ValueError(f'No column named {name}. Refresh the table.')
        return found

    def expr(self, column, alias):
        dtype = column['type']
        cast = 'INTEGER' if pa.types.is_integer(dtype) or pa.types.is_boolean(dtype) else 'REAL' if pa.types.is_floating(dtype) else 'TEXT'
        return f'CAST({alias}.v AS {cast})'

    def ensure(self, name):
        col = self.column(name)
        i = col['id']
        if i in self.loaded:
            return col
        self.db.execute(f'CREATE TABLE c{i}(r INTEGER PRIMARY KEY,v)')
        for start, size, arrays in self.source.batches([col['original']]):
            values = arrays[col['original']].to_pylist()
            values = [v.isoformat() if hasattr(v, 'isoformat') else v for v in values]
            self.db.executemany(f'INSERT INTO c{i} VALUES(?,?)', ((start + j, v) for j, v in enumerate(values)))
        self.db.execute(f'UPDATE c{i} SET v=(SELECT v FROM label_edits WHERE c=? AND r=c{i}.r) WHERE r IN (SELECT r FROM label_edits WHERE c=?)', (col['original'], col['original']))
        self.db.commit()
        self.loaded.add(i)
        return col

    def invalidate(self):
        for name, _ in self.views.values():
            if name:
                self.db.execute(f'DROP TABLE IF EXISTS {name}')
        self.views.clear()
        self.revision += 1

    def view(self, roi_column='', roi_value=None, filters=None, sort=None):
        filters = deepcopy(filters or [])
        if roi_column and roi_value is not None:
            filters.insert(0, {'col': roi_column, 'values': [str(roi_value)]})
        key = json.dumps([filters, sort], sort_keys=True)
        if key in self.views:
            self.views.move_to_end(key)
            return self.views[key]
        if not filters and not sort and not self.db.execute('SELECT 1 FROM deleted LIMIT 1').fetchone():
            return None, self.source.length
        names = list(dict.fromkeys([f['col'] for f in filters] + ([sort['col']] if sort else []))) or [self.names[0]]
        cols = [self.ensure(n) for n in names]
        aliases = {c['name']: f'a{i}' for i, c in enumerate(cols)}
        joined = f'c{cols[0]["id"]} a0' + ''.join(f' JOIN c{c["id"]} a{i} ON a{i}.r=a0.r' for i, c in enumerate(cols[1:], 1))
        clauses = ['NOT EXISTS(SELECT 1 FROM deleted d WHERE d.r=a0.r)'];params = []
        for item in filters:
            col = self.column(item['col']);value = self.expr(col, aliases[item['col']])
            if 'values' in item:
                vals = [scalar(v, col['type'])[0] for v in item['values']]
                # JSON avoids SQLite's variable limit for large picked-cell sets.
                clauses.append(f'{value} IN (SELECT value FROM json_each(?))');params.append(json.dumps(vals))
            elif 'contains' in item:
                clauses.append(f'instr(lower(CAST({value} AS TEXT)),lower(?))>0');params.append(str(item['contains']))
            else:
                clauses.append(f'{value} IS NOT NULL')
                for key_, operator in [('lo','>='),('hi','<=')]:
                    if item.get(key_) is not None:
                        clauses.append(f'CAST({value} AS REAL){operator}?');params.append(float(item[key_]))
        order = 'a0.r'
        if sort:
            expr = self.expr(self.column(sort['col']), aliases[sort['col']])
            order = f'{expr} IS NULL,{expr} '+('DESC' if sort.get('desc') else 'ASC')+',a0.r'
        self.serial += 1;name = f'v{self.serial}'
        self.db.execute(f'CREATE TABLE {name}(seq INTEGER PRIMARY KEY,r INTEGER UNIQUE)')
        self.db.execute(f'INSERT INTO {name}(r) SELECT a0.r FROM {joined} WHERE '+ ' AND '.join(clauses)+f' ORDER BY {order}',params)
        count = self.db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
        self.db.commit();self.views[key] = (name,count)
        while len(self.views)>4:
            old,_ = self.views.popitem(last=False)[1]
            if old:self.db.execute(f'DROP TABLE {old}')
        return name,count

    def ids(self, view, offset, limit):
        name, count = view
        return list(range(offset,min(count,offset+limit))) if name is None else [r[0] for r in self.db.execute(f'SELECT r FROM {name} WHERE seq>? AND seq<=? ORDER BY seq',(offset,offset+limit))]

    def take(self, ids, names):
        cols = [self.column(n) for n in names]
        untouched = [c['original'] for c in cols if c['id'] not in self.loaded]
        raw = self.source.take(ids, untouched)
        return self.overlay(ids,cols,raw)

    def overlay(self,ids,cols,raw):
        result = {}
        for col in cols:
            if col['id'] in self.loaded:
                values = dict(self.db.execute(f'SELECT r,v FROM c{col["id"]} WHERE r IN (SELECT value FROM json_each(?))',(json.dumps(ids),)))
                array = arrow([values[r] for r in ids],col['type'])
            else:
                values = raw[col['original']]
                positions = {r:i for i,r in enumerate(ids)}
                if col['original'] in self.label_columns:
                    for r,v in self.db.execute('SELECT r,v FROM label_edits WHERE c=? AND r IN (SELECT value FROM json_each(?))',(col['original'],json.dumps(ids))):
                        values[positions[r]]=v
                array = arrow(values,col['type'])
            result[col['name']] = array.to_pylist()
        return result

    def state(self):
        return {'revision':self.revision,'edited':self.cursor,'can_undo':self.cursor>0,'can_redo':self.cursor<len(self.history),
                'columns':self.names,'kinds':{c['name']:kind(c['type']) for c in self.columns},'progress':self.progress,
                'renamed':{c['original']:c['name'] for c in self.columns if c['original']!=c['name']},
                'removed':[n for n in self.source.fields if n not in {c['original'] for c in self.columns}]}

    def mutate(self, op, roi_column='', roi_value=None, filters=None, selected=None, revision=None):
        self.check()
        if revision is not None and int(revision)!=self.revision:
            raise ValueError('The table changed since this page was loaded. Reload before editing.')
        operation=op.get('kind');before=deepcopy(self.columns);h=self.serial+1;self.serial=h
        if operation not in ('cells','replace','fill','delete_rows','rename_column','delete_columns'):
            raise ValueError('Choose an editing action.')
        affected=0
        try:
            if operation=='rename_column':
                col=self.column(op['column']);name=str(op.get('name','')).strip()
                if not name or '\x00' in name or len(name)>256:raise ValueError('Enter a column name (up to 256 characters).')
                if name in self.names and name!=col['name']:raise ValueError('That column name already exists.')
                if name==col['name']:return {**self.state(),'affected':0}
                col['name']=name;affected=1
            elif operation=='delete_columns':
                names=set(op.get('columns') or [])
                for n in names:self.column(n)
                if len(names)>=len(self.columns):raise ValueError('Keep at least one column.')
                self.columns=[c for c in self.columns if c['name'] not in names];affected=len(names)
            else:
                if operation=='cells':
                    changes=op.get('changes') or []
                    if len(changes)>10000:raise ValueError('Paste at most 10,000 values at once; use Fill or Replace for larger edits.')
                    # Load columns before starting the edit transaction.
                    for name in dict.fromkeys(c['col'] for c in changes):self.ensure(name)
                    self.db.execute('SAVEPOINT mutation')
                    for item in changes:
                        row=int(item['row']);col=self.column(item['col']);table=f'c{col["id"]}'
                        if not 0<=row<self.source.length or self.db.execute('SELECT 1 FROM deleted WHERE r=?',(row,)).fetchone():raise ValueError('That row is no longer available.')
                        new,dtype=scalar(item.get('value'),col['type']);col['type']=dtype
                        old=self.db.execute(f'SELECT v FROM {table} WHERE r=?',(row,)).fetchone()[0]
                        self.db.execute('INSERT INTO changes VALUES(?,?,?,?,?) ON CONFLICT(h,c,r) DO UPDATE SET new=excluded.new',(h,col['id'],row,old,new))
                        self.db.execute(f'UPDATE {table} SET v=? WHERE r=?',(new,row))
                    affected=self.db.execute('SELECT COUNT(*) FROM changes WHERE h=?',(h,)).fetchone()[0]
                else:
                    scope=op.get('scope','filtered')
                    if scope not in ('all','filtered','selected'):raise ValueError('Choose the affected rows.')
                    col=self.ensure(op['column']) if operation!='delete_rows' else None
                    view=self.view('' if scope=='all' else roi_column,None if scope=='all' else roi_value,[] if scope=='all' else filters,None)
                    # Reuse disk-backed view IDs. All rows use a lazily loaded anchor column.
                    if view[0] is None:
                        anchor=self.ensure(self.names[0]);selection=f'SELECT r FROM c{anchor["id"]}'
                    else:selection=f'SELECT r FROM {view[0]}'
                    if scope=='selected':
                        ids=list(dict.fromkeys(int(r) for r in selected or []))
                        if not ids:raise ValueError('Select rows first.')
                        selection+=' WHERE r IN (SELECT value FROM json_each('+"'"+json.dumps(ids)+"'"+'))'
                    self.db.execute('SAVEPOINT mutation')
                    if operation=='delete_rows':
                        self.db.execute(f'INSERT INTO changes(h,c,r,old,new) SELECT ?, -1,r,0,1 FROM ({selection})',(h,))
                        self.db.execute('INSERT OR IGNORE INTO deleted SELECT r FROM changes WHERE h=? AND c=-1',(h,))
                    else:
                        new,dtype=scalar(op.get('value'),col['type'])
                        table=f'c{col["id"]}';extra='';params=[h,col['id'],new]
                        if operation=='replace':
                            old,_=scalar(op.get('old'),col['type']);extra=' AND '+self.expr(col,table)+' IS ?';params.append(old)
                        self.db.execute(f'INSERT INTO changes SELECT ?,?,r,v,? FROM {table} WHERE r IN ({selection})'+extra,params)
                        self.db.execute(f'UPDATE {table} SET v=(SELECT new FROM changes WHERE h=? AND c=? AND r={table}.r) WHERE r IN (SELECT r FROM changes WHERE h=? AND c=?)',(h,col['id'],h,col['id']))
                        col['type']=dtype
                    affected=self.db.execute('SELECT COUNT(*) FROM changes WHERE h=?',(h,)).fetchone()[0]
                self.db.execute('RELEASE mutation')
            if not affected:
                self.columns=before;return {**self.state(),'affected':0}
            for item in self.history[self.cursor:]:self.db.execute('DELETE FROM changes WHERE h=?',(item['id'],))
            self.history=self.history[:self.cursor]+[{'id':h,'before':before,'after':deepcopy(self.columns),'kind':operation,'affected':affected}]
            self.cursor+=1;self.invalidate();self.db.commit()
            return {**self.state(),'affected':affected}
        except Exception:
            self.db.rollback();self.columns=before
            raise

    def undo(self, redo=False):
        self.check()
        if redo and self.cursor>=len(self.history) or not redo and not self.cursor:return self.state()
        item=self.history[self.cursor if redo else self.cursor-1];h=item['id'];value='new' if redo else 'old'
        for (c,) in self.db.execute('SELECT DISTINCT c FROM changes WHERE h=?',(h,)).fetchall():
            if c==-1:
                if redo:self.db.execute('INSERT OR IGNORE INTO deleted SELECT r FROM changes WHERE h=? AND c=-1',(h,))
                else:self.db.execute('DELETE FROM deleted WHERE r IN (SELECT r FROM changes WHERE h=? AND c=-1)',(h,))
            else:self.db.execute(f'UPDATE c{c} SET v=(SELECT {value} FROM changes WHERE h=? AND c=? AND r=c{c}.r) WHERE r IN (SELECT r FROM changes WHERE h=? AND c=?)',(h,c,h,c))
        self.columns=deepcopy(item['after' if redo else 'before']);self.cursor+=1 if redo else -1
        self.invalidate();self.db.commit();return self.state()

    def close(self):
        if not self.closed:
            self.closed = True
            self.db.close();self.directory.cleanup()


def editor(path, discard=False):
    key=str(Path(path).resolve())
    with LOCK:
        item=SESSIONS.get(key)
        if item and (discard or stamp(key)!=item.stamp):
            if not discard and item.cursor: item.check()
            if not item.lock.acquire(blocking=False):
                raise ValueError('A Data operation is still running. Try again when it finishes.')
            try:item.close()
            finally:item.lock.release()
            del SESSIONS[key];item=None
        if item is None:
            item=SESSIONS[key]=Editor(key)
        SESSIONS.move_to_end(key)
        for old in list(SESSIONS):
            if len(SESSIONS)<=4:break
            if old!=key and not SESSIONS[old].cursor and SESSIONS[old].lock.acquire(blocking=False):
                try:SESSIONS[old].close()
                finally:SESSIONS[old].lock.release()
                del SESSIONS[old]
        return item


@contextmanager
def locked(path):
    item=editor(path)
    with item.lock:
        item.check();yield item


def info(path, column=''):
    with locked(path) as e:
        guess=column if column in e.names else next((n for n in e.names if n.lower() in ('batch_id','roi','image','sample','image_id','roi_id')),'')
        values=[]
        if guess:
            c=e.ensure(guess);expr=e.expr(c,'a')
            values=[str(r[0]) for r in e.db.execute(f'SELECT DISTINCT {expr} FROM c{c["id"]} a WHERE v IS NOT NULL ORDER BY {expr} LIMIT 5000')]
        return {**e.state(),'name':Path(path).name,'image_column':guess,'values':values,'rows':e.source.length-e.db.execute('SELECT COUNT(*) FROM deleted').fetchone()[0]}


def query(path,roi_column='',roi_value=None,filters=None,sort=None,offset=0,limit=200,columns=None):
    from .data_view import _cell
    with locked(path) as e:
        view=e.view(roi_column,roi_value,filters,sort);total=e.view(roi_column,roi_value,[],None)[1]
        offset=max(0,min(int(offset or 0),max(0,view[1]-1)));limit=max(1,min(int(limit or 200),1000))
        names=[n for n in columns or [] if n in e.names] or e.names
        ids=e.ids(view,offset,limit);values=e.take(ids,names)
        return {**e.state(),'shown':names,'rows':[[r,*[_cell(values[n][i]) for n in names]] for i,r in enumerate(ids)],
                'matched':view[1],'total':total,'offset':offset,'label':'label' in e.names}


def summary(path,roi_column='',roi_value=None,filters=None,column='',limit=30,bins=24):
    with locked(path) as e:
        col=e.ensure(column);view=e.view(roi_column,roi_value,filters,None);expr=e.expr(col,'a')
        joined=f'c{col["id"]} a'+(f' JOIN {view[0]} s ON s.r=a.r' if view[0] else '')
        base=f'SELECT {expr} v FROM {joined}'
        if kind(col['type'])=='group':
            groups=e.db.execute(f'SELECT v,COUNT(*) FROM ({base}) WHERE v IS NOT NULL AND v!=? GROUP BY v ORDER BY COUNT(*) DESC,v LIMIT ?',( '',max(1,min(int(limit),5000)))).fetchall()
            distinct,nonempty=e.db.execute(f'SELECT COUNT(DISTINCT v),COUNT(v) FROM ({base}) WHERE v!=?',('',)).fetchone()
            return {'column':column,'kind':'group','count':view[1],'empty':view[1]-nonempty,'groups':groups,'distinct':distinct}
        count,low,high,mean=e.db.execute(f'SELECT COUNT(v),MIN(v),MAX(v),AVG(v) FROM ({base})').fetchone()
        result={'column':column,'kind':'value','count':view[1],'empty':view[1]-count}
        if count:
            middle=[r[0] for r in e.db.execute(f'SELECT v FROM ({base}) WHERE v IS NOT NULL ORDER BY v LIMIT ? OFFSET ?',(2 if count%2==0 else 1,(count-1)//2))]
            edges=np.linspace(low if high!=low else low-.5,high if high!=low else high+.5,max(1,min(int(bins),200))+1)
            counts=[0]*(len(edges)-1)
            width=edges[-1]-edges[0]
            for bin_,n in e.db.execute(f'SELECT MIN(?,MAX(0,CAST((v-?)/?*? AS INTEGER))),COUNT(*) FROM ({base}) WHERE v IS NOT NULL GROUP BY 1',(len(counts)-1,float(edges[0]),float(width),len(counts))):counts[bin_]=n
            result.update(min=low,max=high,mean=mean,median=sum(middle)/len(middle),bins=counts,edges=edges.tolist())
        return result


def edit(path, changes):
    with locked(path) as e:return e.mutate({'kind':'cells','changes':changes})['edited']


def mutate(path, op, roi_column='',roi_value=None,filters=None,selected=None,revision=None):
    with locked(path) as e:return e.mutate(op,roi_column,roi_value,filters,selected,revision)


def history(path,action):
    if action=='discard':return editor(path,discard=True).state()
    with locked(path) as e:return e.undo(action=='redo')


def values_at_rows(path,ids,column):
    with locked(path) as e:return e.take([int(i) for i in ids],[column])[column]


def current_name(path,name):
    with locked(path) as e:return next((c['name'] for c in e.columns if c['original']==name),name)


def find(path,roi_column,roi_value,filters,sort,label,label_column='label',identity_roi_column='',identity_roi=None):
    with locked(path) as e:
        if label_column not in e.names:return None
        view=e.view(roi_column,roi_value,filters,sort)
        identity=[{'col':label_column,'values':[label]}]
        if identity_roi is not None and identity_roi_column in e.names:identity.append({'col':identity_roi_column,'values':[identity_roi]})
        matches=e.view(roi_column,roi_value,[*(filters or []),*identity],None)
        if not matches[1]:return None
        row=e.ids(matches,0,1)[0]
        return e.db.execute(f'SELECT seq-1 FROM {view[0]} WHERE r=?',(row,)).fetchone()[0] if view[0] else row


def labels_of(path,roi_column,roi_value,filters,label_column='label'):
    with locked(path) as e:
        if label_column not in e.names:return []
        view=e.view(roi_column,roi_value,filters,None);col=e.ensure(label_column)
        joined=f'c{col["id"]} a'+(f' JOIN {view[0]} s ON s.r=a.r' if view[0] else '')
        return [int(r[0]) for r in e.db.execute(f'SELECT DISTINCT v FROM {joined} WHERE v IS NOT NULL')]


def export_chunks(path,roi_column='',roi_value=None,filters=None,sort=None,columns=None):
    import io,csv
    e=editor(path)
    with e.lock:
        e.check();view=e.view(roi_column,roi_value,filters,sort);revision=e.revision
        names=[n for n in columns or [] if n in e.names] or e.names
    buffer=io.StringIO();writer=csv.writer(buffer,lineterminator='\n');writer.writerow(names);yield buffer.getvalue()
    for offset in range(0,view[1],BATCH):
        with e.lock:
            e.check()
            if revision!=e.revision:raise ValueError('The table changed during export. Export again.')
            ids=e.ids(view,offset,BATCH);values=e.take(ids,names)
        buffer=io.StringIO();writer=csv.writer(buffer,lineterminator='\n');writer.writerows(zip(*(values[n] for n in names)));yield buffer.getvalue()


def export(path,roi_column='',roi_value=None,filters=None,sort=None,columns=None):
    return ''.join(export_chunks(path,roi_column,roi_value,filters,sort,columns))


def save(path,mode='copy',folder=None,name=None,revision=None):
    """Atomic streamed save, with backup and all saved phenotype overlays included."""
    from spacemap import labels,pending_columns,dataio
    import pyarrow.csv as pacsv
    with locked(path) as e,labels.write_lock(path):
        e.check()
        if revision is not None and int(revision)!=e.revision:raise ValueError('The table changed since you reviewed it. Reload before saving.')
        if pending_columns.has_stale(path):raise ValueError('Some saved result columns are stale. Reapply them by cell ID or discard them before saving this table.')
        if mode not in ('copy','overwrite'):raise ValueError('Choose Save or Save as.')
        source=Path(path).resolve()
        if mode=='overwrite':target=source
        else:
            filename=str(name or source.stem+'_edited'+source.suffix).strip()
            if Path(filename).name!=filename or filename in ('','.','..'):raise ValueError('Enter a file name, without a directory.')
            if Path(filename).suffix.lower() not in ('.csv','.tsv','.parquet'):raise ValueError('Use .csv, .tsv or .parquet.')
            target=(Path(folder) if folder else source.parent)/filename
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if name:raise ValueError('That file already exists. Choose a different name.')
                target=target.with_name(target.stem+'_'+uuid.uuid4().hex[:8]+target.suffix)
        temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
        schema=pa.schema([pa.field(c['name'],c['type']) for c in e.columns]);written=0
        e.progress={'state':'saving','done':0,'total':e.source.length}
        try:
            writer=(pq.ParquetWriter(temporary,schema,compression='snappy') if target.suffix=='.parquet' else
                    pacsv.CSVWriter(temporary,schema,write_options=pacsv.WriteOptions(delimiter='\t' if target.suffix=='.tsv' else ',')))
            try:
                for start,size,arrays in e.source.batches([c['original'] for c in e.columns]):
                    ids=list(range(start,start+size));values=e.overlay(ids,e.columns,{n:a.to_pylist() for n,a in arrays.items()})
                    deleted={r[0] for r in e.db.execute('SELECT r FROM deleted WHERE r>=? AND r<?',(start,start+size))}
                    keep=pa.array([r not in deleted for r in ids])
                    batch=pa.Table.from_arrays([arrow(values[c['name']],c['type']) for c in e.columns],schema=schema).filter(keep)
                    writer.write_table(batch);written+=batch.num_rows
                    e.progress={'state':'saving','done':start+size,'total':e.source.length}
            finally:writer.close()
            e.check()
            if mode=='overwrite':
                backup=source.parent/'previous'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);backup.mkdir(parents=True)
                shutil.copy2(source,backup/source.name)
                sidecar=labels.sidecar_path(source)
                if sidecar.exists():shutil.copy2(sidecar,backup/sidecar.name)
                pending=pending_columns.store_dir(source)
                if pending.is_dir():shutil.copytree(pending,backup/pending.name)
            e.check()
            os.replace(temporary,target)
            if mode=='overwrite':
                # Overlay fingerprints become stale on replacement. Remove only the
                # materialized versions, preserved in the backup alongside the original.
                labels.clear(source);pending_columns.discard(source)
                if source.suffix.lower()!='.parquet':dataio.clear_fast_copy(source)
            e.progress={'state':'saved','done':e.source.length,'total':e.source.length,'rows':written,'path':str(target)}
            with LOCK:SESSIONS.pop(e.path,None)
            e.close()
            return str(target)
        except Exception as error:
            temporary.unlink(missing_ok=True);e.progress={'state':'error','message':str(error)};raise


def status(path):
    item=SESSIONS.get(str(Path(path).resolve()))
    return {'progress':item.progress if item else None}


def export_file(path,roi_column='',roi_value=None,filters=None,sort=None,columns=None):
    # Finish a consistent export on disk before sending it to the browser.
    with locked(path) as e:
        handle=tempfile.NamedTemporaryFile(prefix='spacemap-export-',suffix='.csv',delete=False,mode='w',encoding='utf-8',newline='')
        try:
            with handle:
                for chunk in export_chunks(path,roi_column,roi_value,filters,sort,columns):handle.write(chunk)
            return handle.name
        except Exception:
            Path(handle.name).unlink(missing_ok=True);raise
