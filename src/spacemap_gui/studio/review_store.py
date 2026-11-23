"""Project-local review queue. Decisions are proposals; applying is a separate action.

SQLite uses rollback journals, never WAL (projects may live on network storage).
Every connection is covered by the project's cross-process advisory lock.
The queue is independent of viewer sessions and never rewrites research tables.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

KINDS = {'annotation', 'cells', 'leiden', 'som', 'vortex', 'subcluster',
         'neighbourhood', 'tls', 'gland', 'niche', 'gland_type', 'niche_type'}
DECISIONS = {'keep', 'relabel', 'reject', 'unable'}


def home(root):
    return Path(root) / 'outputs' / 'studio' / 'review'


@contextmanager
def database(root):
    import fcntl
    from spacemap.sharing import open_lock
    folder = home(root)
    folder.mkdir(parents=True, exist_ok=True)
    with open_lock(folder / 'queue.lock') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        db = sqlite3.connect(folder / 'queue.sqlite3', timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('PRAGMA synchronous=FULL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY, identity TEXT UNIQUE NOT NULL,
                    kind TEXT NOT NULL, image TEXT NOT NULL, title TEXT NOT NULL,
                    target TEXT NOT NULL, suggestions TEXT NOT NULL, note TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending', version INTEGER NOT NULL DEFAULT 1,
                    decision TEXT, applied TEXT, created REAL NOT NULL, updated REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS items_state ON items(state, kind, image);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, item TEXT, actor TEXT NOT NULL,
                    action TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS grants (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, secret_hash TEXT NOT NULL,
                    scope TEXT NOT NULL, expires REAL NOT NULL, revoked INTEGER DEFAULT 0,
                    created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS suggestion_sets (
                    context TEXT PRIMARY KEY, choices TEXT NOT NULL, updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS polygon_sizes (item TEXT PRIMARY KEY, area REAL);
            ''')
            if 'deleted' not in {r['name'] for r in db.execute('PRAGMA table_info(grants)')}:
                db.execute('ALTER TABLE grants ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0')
            with db:
                yield db
        finally:
            db.close()
            fcntl.flock(handle, fcntl.LOCK_UN)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _event(db, item, actor, action, payload):
    db.execute('INSERT INTO events(item,actor,action,payload,created) VALUES(?,?,?,?,?)',
               (item, actor, action, _json(payload), time.time()))


def _item(row):
    result = dict(row)
    for key in ('target', 'suggestions', 'decision', 'applied'):
        result[key] = json.loads(result[key]) if result[key] else None
    result.pop('identity', None)
    return result


def add(root, **values):
    with database(root) as db:
        item = _add(db, **values)
        if not values.get('requested_by'):
            _send_default(db, [item['id']])
        return item


def add_many(root, items):
    with database(root) as db:
        found = [_add(db, **values) for values in items]
        _send_default(db, [item['id'] for item, values in zip(found, items) if not values.get('requested_by')])
        return found


def _add(db, *, kind, image, title, target, suggestions=(), note='', requested_by=None, area=None):
    if kind not in KINDS:
        raise ValueError('Choose a supported review type.')
    if not isinstance(target, dict) or not target.get('source') or not target.get('ids'):
        raise ValueError('A review needs a saved source and exact object IDs.')
    if not target.get('revision'):
        raise ValueError('Record the source revision before flagging it.')
    suggestions = list(dict.fromkeys(str(v).strip() for v in suggestions if str(v).strip()))[:50]
    identity = hashlib.sha256(_json([kind, str(image), target]).encode()).hexdigest()
    now = time.time()
    old = db.execute('SELECT * FROM items WHERE identity=?', (identity,)).fetchone()
    if old:
        if area is not None:
            db.execute('INSERT OR REPLACE INTO polygon_sizes(item,area) VALUES(?,?)', (old['id'], float(area)))
        return _item(old)
    item = uuid.uuid4().hex
    db.execute('INSERT INTO items(id,identity,kind,image,title,target,suggestions,note,created,updated) '
               'VALUES(?,?,?,?,?,?,?,?,?,?)',
               (item, identity, kind, str(image), str(title)[:500], _json(target),
                _json(suggestions), str(note)[:8000], now, now))
    if area is not None:
        db.execute('INSERT OR REPLACE INTO polygon_sizes(item,area) VALUES(?,?)', (item, float(area)))
    if requested_by:
        db.execute("UPDATE items SET state='requested' WHERE id=?", (item,))
    _event(db, item, requested_by or 'owner', 'nominate' if requested_by else 'flag', {'kind': kind, 'title': title})
    return _item(db.execute('SELECT * FROM items WHERE id=?', (item,)).fetchone())


def get(root, item_id):
    with database(root) as db:
        row = db.execute('SELECT * FROM items WHERE id=?', (item_id,)).fetchone()
        if not row:
            raise ValueError('This review item is no longer available.')
        return _item(row)


def suggestion_context(item):
    target = item['target']
    name = target.get('annotation_name') or (item.get('title', '').split(' · ')[0] if item['kind'] == 'annotation' else target.get('column')) or ''
    return _json([item['kind'], name])


def remembered_suggestions(root):
    if not (home(root) / 'queue.sqlite3').exists():
        return {}
    with database(root) as db:
        return {r['context']: json.loads(r['choices']) for r in db.execute('SELECT * FROM suggestion_sets')}


def suggestion_items_in_images(root, images, kind):
    """Preview all pending flags of one kind in exact selected images, unpaged."""
    if not isinstance(images, list) or not images or len(images) > 5000 or any(not isinstance(v, str) or not v for v in images):
        raise ValueError('Select images first.')
    if kind not in KINDS:
        raise ValueError('Choose one review type.')
    paths = {str(Path(value).resolve()) for value in images}
    with database(root) as db:
        rows = [_item(row) for row in db.execute('SELECT * FROM items WHERE kind=? ORDER BY created,id', (kind,))]
        matched = [row for row in rows if row['target'].get('image') and str(Path(row['target']['image']).resolve()) in paths]
        items = [row for row in matched if row['state'] == 'pending']
        return {'items': items, 'skipped': len(matched)-len(items)}


def item_ids_in_images(root, images, kind=''):
    """All existing flags in exact chosen images, independently of queue paging.

    Sharing images alone is still supported. Call this only when the owner
    explicitly includes their review items in the Share dialog.
    """
    if kind and kind not in KINDS:
        raise ValueError('Choose a supported review type.')
    paths = {str(Path(path).resolve()) for path in images}
    if not paths or not (home(root) / 'queue.sqlite3').exists():
        return []
    with database(root) as db:
        rows = db.execute("SELECT id,json_extract(target,'$.image') AS path FROM items WHERE state!='requested'" +
                          (' AND kind=?' if kind else '') + ' ORDER BY created,id',
                          (kind,) if kind else ()).fetchall()
        # Resolve each distinct path once, not once per polygon.
        matches = {path for path in {row['path'] for row in rows} if path and str(Path(path).resolve()) in paths}
        ids = [row['id'] for row in rows if row['path'] in matches]
        if len(ids) > 20000:
            raise ValueError('Choose fewer images or one review type to share at most 20,000 items.')
        return ids


def set_suggestions(root, items, choices, *, note=None, remember=False):
    """One atomic metadata update, with no source edits or changes to decisions."""
    if not isinstance(items, dict) or not items:
        raise ValueError('Select review items first.')
    if not isinstance(choices, list) or len(choices) > 50 or any(not isinstance(v, str) or len(v) > 200 for v in choices):
        raise ValueError('Choose up to 50 short suggestions.')
    choices = list(dict.fromkeys(v.strip() for v in choices if v.strip()))
    with database(root) as db:
        rows = list(db.execute('SELECT * FROM items WHERE id IN (SELECT value FROM json_each(?))', (_json(list(items)),)))
        if len(rows) != len(items) or any(r['version'] != items[r['id']] for r in rows):
            raise ValueError('These items changed. Reopen Suggestions before saving.')
        if any(r['state'] in ('applying', 'applied') for r in rows):
            raise ValueError('Reopen completed items before changing their suggestions.')
        for row in rows:
            db.execute('UPDATE items SET suggestions=?,note=?,version=version+1,updated=? WHERE id=?',
                       (_json(choices), row['note'] if note is None else str(note)[:8000], time.time(), row['id']))
            _event(db, row['id'], 'owner', 'suggestions', {'choices': choices, 'remember': bool(remember)})
            if remember:
                db.execute('INSERT OR REPLACE INTO suggestion_sets(context,choices,updated) VALUES(?,?,?)',
                           (suggestion_context(_item(row)), _json(choices), time.time()))
        return {'updated': len(rows), 'suggestions': choices}


def list_items(root, *, kind='', state='', search='', offset=0, limit=100, ids=None, ids_only=False,
               since=-1, grant_id=None, sort='created', image=None):
    if not (home(root) / 'queue.sqlite3').exists():
        return {'items': [], 'ids': [], 'total': 0, 'revision': 0, 'counts': [], 'removed': []}
    clauses, args = [], []
    if image is not None:
        clauses.append("json_extract(target,'$.image')=?"); args.append(image)
    for column, value in (('kind', kind), ('state', state)):
        if value:
            clauses.append(column + '=?'); args.append(value)
    if search:
        clauses.append('(instr(lower(title),?) OR instr(lower(image),?))')
        args.extend([str(search).lower()] * 2)
    if ids is not None:
        ids = list(dict.fromkeys(ids))
        clauses.append('id IN (SELECT value FROM json_each(?))'); args.append(_json(ids))
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    with database(root) as db:
        if ids_only:
            rows = db.execute('SELECT id FROM items' + where + ' ORDER BY created,id LIMIT 20001', args).fetchall()
            if len(rows) > 20000:
                raise ValueError('Narrow the search to select at most 20,000 items at once.')
            return {'ids': [r['id'] for r in rows],
                    'revision': db.execute('SELECT coalesce(max(seq),0) FROM events').fetchone()[0]}
        order = {
            'created': 'created,id',
            'size_down': 'area IS NULL,area DESC,created,id',
            'size_up': 'area IS NULL,area,created,id',
            'place': "image,json_extract(target,'$.bounds[1]'),json_extract(target,'$.bounds[0]'),created,id",
        }.get(sort)
        if order is None:
            raise ValueError('Choose a supported polygon sort order.')
        rows = db.execute('SELECT *, (SELECT area FROM polygon_sizes WHERE item=items.id) AS area FROM items' + where + ' ORDER BY ' + order + ' LIMIT ? OFFSET ?',
                          (*args, max(1, min(200, int(limit))), max(0, int(offset)))).fetchall()
        counts = db.execute('SELECT image,kind,state,count(*) AS n FROM items' + where +
                            ' GROUP BY image,kind,state', args).fetchall()
        removed = []
        if int(since) >= 0:
            # Removed items no longer belong to a grant's current scope. Check
            # the scope recorded at removal, without revealing other queues.
            removal_rows = db.execute("SELECT item,payload FROM events WHERE action='remove' AND seq>?", (int(since),))
            removed = [r['item'] for r in removal_rows if not grant_id or
                       grant_id in json.loads(r['payload']).get('grants', [])]
        return {'items': [_item(r) for r in rows], 'total': sum(r['n'] for r in counts), 'removed': removed,
                'counts': [dict(r) for r in counts],
                'revision': db.execute('SELECT coalesce(max(seq),0) FROM events').fetchone()[0]}


def remove(root, ids):
    """Remove owner-selected review entries, never their source objects or edits.

    Retain decisions and the complete entry in the audit history; removing an
    applied review does not undo it. Queue and shared-link scopes change together.
    """
    if not isinstance(ids, list) or not ids or len(ids) > 20000 or any(not isinstance(v, str) for v in ids):
        raise ValueError('Select up to 20,000 review items to remove.')
    ids = list(dict.fromkeys(ids))
    with database(root) as db:
        rows = list(db.execute('SELECT * FROM items WHERE id IN (SELECT value FROM json_each(?))', (_json(ids),)))
        if len(rows) != len(ids):
            raise ValueError('Some review items changed. Refresh the list before removing them.')
        if any(r['state'] == 'applying' for r in rows):
            raise ValueError('Wait for the current approval to finish before removing these items.')
        removed = set(ids)
        shared_with = {item: [] for item in ids}
        for grant in list(db.execute('SELECT id,scope FROM grants')):
            scope = json.loads(grant['scope'])
            affected = removed.intersection(scope['items'])
            if not affected:
                continue
            for item in affected:
                shared_with[item].append(grant['id'])
            scope['items'] = [item for item in scope['items'] if item not in removed]
            db.execute('UPDATE grants SET scope=? WHERE id=?', (_json(scope), grant['id']))
        for row in rows:
            _event(db, row['id'], 'owner', 'remove', {'entry': _item(row), 'grants': shared_with[row['id']]})
        db.execute('DELETE FROM items WHERE id IN (SELECT value FROM json_each(?))', (_json(ids),))
        db.execute('DELETE FROM polygon_sizes WHERE item IN (SELECT value FROM json_each(?))', (_json(ids),))
    return {'removed': ids}


def missing_polygon_sizes(root, ids=None):
    with database(root) as db:
        return [_item(r) for r in db.execute(
            "SELECT * FROM items WHERE kind='annotation' AND id NOT IN (SELECT item FROM polygon_sizes)" +
            (' AND id IN (SELECT value FROM json_each(?))' if ids is not None else ''),
            (_json(list(ids)),) if ids is not None else ())]


def save_polygon_sizes(root, sizes):
    # Derived metadata only: never alter target identity, decisions or versions.
    with database(root) as db:
        db.executemany('INSERT OR REPLACE INTO polygon_sizes(item,area) SELECT ?,? WHERE EXISTS(SELECT 1 FROM items WHERE id=?)',
                       [(item, area, item) for item, area in sizes.items()])


def decide(root, item_id, *, version, action, label='', note='', actor='owner', grant=None):
    return decide_many(root, {item_id: version}, action=action, label=label, note=note, actor=actor, grant=grant)[0]


def decide_many(root, items, *, action, label='', note=None, actor='owner', grant=None):
    """Record one decision for a selection atomically; never partly update stale selections."""
    if not isinstance(items, dict) or not items or len(items) > 20000:
        raise ValueError('Select up to 20,000 review items.')
    if action not in DECISIONS:
        raise ValueError('Choose Keep, Relabel, Reject or Unable to decide.')
    label = str(label).strip()
    if action == 'relabel' and not label:
        raise ValueError('Enter the suggested label.')
    with database(root) as db:
        if grant:
            allowed = _authorize(db, *grant)
            if not set(items).issubset(allowed['scope']['items']):
                raise PermissionError('These items were not shared with this link.')
            actor = allowed['id']
        rows = []
        for item_id, version in items.items():
            row = db.execute('SELECT * FROM items WHERE id=?', (item_id,)).fetchone()
            if not row or row['version'] != version:
                raise ValueError('This item changed. Reload its latest decision first.')
            if row['state'] in ('applying', 'applied', 'requested'):
                raise ValueError('The owner must reopen this item before another decision.')
            rows.append(row)
        results = []
        for row in rows:
            item_id = row['id']
            previous = json.loads(row['decision']) if row['decision'] else {}
            comment = previous.get('note', '') if note is None else str(note).strip()
            decision = {'action': action, 'label': label[:500], 'note': comment[:8000], 'actor': actor,
                        'at': datetime.now(timezone.utc).isoformat()}
            db.execute('UPDATE items SET decision=?,state=?,version=version+1,updated=? WHERE id=?',
                       (_json(decision), 'unresolved' if action == 'unable' else 'answered', time.time(), item_id))
            _event(db, item_id, actor, 'decision', decision)
            results.append(_item(db.execute('SELECT * FROM items WHERE id=?', (item_id,)).fetchone()))
        return results


def transition(root, item_id, *, version, state, result=None):
    if state not in ('pending', 'applying', 'applied', 'conflict'):
        raise ValueError('Invalid review state.')
    with database(root) as db:
        row = db.execute('SELECT * FROM items WHERE id=?', (item_id,)).fetchone()
        if not row or row['version'] != version:
            raise ValueError('This review changed. Reload before applying it.')
        if state == 'applying' and row['state'] != 'answered':
            raise ValueError('Choose a decision before applying it.')
        if state == 'applied' and row['state'] != 'applying':
            raise ValueError('This decision has not started applying.')
        db.execute('UPDATE items SET state=?,applied=?,version=version+1,updated=? WHERE id=?',
                   (state, _json(result) if result else None, time.time(), item_id))
        _event(db, item_id, 'owner', state, result or {})
        return _item(db.execute('SELECT * FROM items WHERE id=?', (item_id,)).fetchone())


def attach_application_job(root, item_id, version, result):
    """Keep a long approval applying until its associated job actually succeeds."""
    with database(root) as db:
        row=db.execute('SELECT * FROM items WHERE id=?',(item_id,)).fetchone()
        if not row or row['version']!=version or row['state']!='applying':
            raise ValueError('This review changed while starting its update.')
        db.execute('UPDATE items SET applied=?,version=version+1,updated=? WHERE id=?',
                   (_json(result),time.time(),item_id))
        _event(db,item_id,'owner','application_job',result)
        return _item(db.execute('SELECT * FROM items WHERE id=?',(item_id,)).fetchone())


def history(root, item_id):
    with database(root) as db:
        return [{**dict(r), 'payload': json.loads(r['payload'])} for r in db.execute(
            'SELECT * FROM events WHERE item=? ORDER BY seq', (item_id,))]


def create_grant(root, *, name, items, context=None, days=14):
    items = list(dict.fromkeys(items))
    if (not items and not (context or {}).get('images')) or len(items) > 20000 or not str(name).strip():
        raise ValueError('Name the link and select images or review items to share (up to 20,000 items).')
    secret, grant_id = secrets.token_urlsafe(32), uuid.uuid4().hex
    with database(root) as db:
        count = db.execute('SELECT count(*) FROM items WHERE id IN (SELECT value FROM json_each(?))',
                           (_json(items),)).fetchone()[0]
        if count != len(items):
            raise ValueError('Some selected items are no longer available.')
        scope = {'items': items, 'context': context or {}}
        db.execute('INSERT INTO grants(id,name,secret_hash,scope,expires,created) VALUES(?,?,?,?,?,?)',
                   (grant_id, str(name).strip()[:200], hashlib.sha256(secret.encode()).hexdigest(),
                    _json(scope), time.time() + min(90, max(1, int(days))) * 86400, time.time()))
        _event(db, None, 'owner', 'share', {'grant': grant_id, 'items': items})
    return {'id': grant_id, 'secret': secret, 'name': name}


def _authorize(db, grant_id, secret):
    row = db.execute('SELECT * FROM grants WHERE id=?', (grant_id,)).fetchone()
    if not row or row['revoked'] or row['expires'] <= time.time() or not secrets.compare_digest(
            row['secret_hash'], hashlib.sha256(str(secret).encode()).hexdigest()):
        raise PermissionError('This review link has expired or was revoked.')
    return {k: json.loads(v) if k == 'scope' else v for k, v in dict(row).items() if k != 'secret_hash'}


def authorize(root, grant_id, secret):
    with database(root) as db:
        return _authorize(db, grant_id, secret)


def grants(root):
    if not (home(root) / 'queue.sqlite3').exists():
        return []
    with database(root) as db:
        return [{**dict(r), 'scope': json.loads(r['scope'])} for r in db.execute(
            'SELECT id,name,scope,expires,revoked,created FROM grants WHERE deleted=0 ORDER BY created DESC')]


def revoke(root, grant_id):
    with database(root) as db:
        db.execute('UPDATE grants SET revoked=1 WHERE id=?', (grant_id,))
        _event(db, None, 'owner', 'revoke', {'grant': grant_id})


def delete_grant(root, grant_id):
    """Disable and hide a link, retaining its decisions and audit history."""
    with database(root) as db:
        row = db.execute('SELECT id,deleted FROM grants WHERE id=?', (grant_id,)).fetchone()
        if not row:
            raise ValueError('This review link no longer exists.')
        db.execute('UPDATE grants SET revoked=1,deleted=1 WHERE id=?', (grant_id,))
        db.execute("DELETE FROM preferences WHERE key='destination' AND value=?", (grant_id,))
        if not row['deleted']:
            _event(db, None, 'owner', 'delete_link', {'grant': grant_id})


def update_grant(root, grant_id, *, name, access, items=None, days=None, context=None):
    """Edit one active share in place; preserve its token, decisions and scope."""
    if not str(name).strip() or access not in ('internal', 'external'):
        raise ValueError('Name the reviewer link and choose its access setting.')
    with database(root) as db:
        row = db.execute('SELECT * FROM grants WHERE id=? AND revoked=0 AND expires>?',
                         (grant_id, time.time())).fetchone()
        if not row:
            raise ValueError('This reviewer link is closed. Create a new link.')
        added = _append(db, grant_id, items) if items else {'added': 0}
        row = db.execute('SELECT * FROM grants WHERE id=?', (grant_id,)).fetchone()
        scope = json.loads(row['scope'])
        scope['access'] = access
        if context is not None:
            previous = scope.setdefault('context', {})
            previous.setdefault('images', {}).update(context.get('images', {}))
            previous.setdefault('cells', {}).update(context.get('cells', {}))
            if 'metadata_columns' in context:
                previous['metadata_columns'] = context['metadata_columns']
                # Removing a shared metadata column removes it from every image.
                for image in previous.get('images', {}).values():
                    image['metadata'] = {k: v for k, v in image.get('metadata', {}).items()
                                         if k in context['metadata_columns']}
        expires = row['expires'] if days is None else time.time() + min(90, max(1, int(days))) * 86400
        db.execute('UPDATE grants SET name=?,scope=?,expires=? WHERE id=?',
                   (str(name).strip()[:200], _json(scope), expires, grant_id))
        _event(db, None, 'owner', 'share_update', {'grant': grant_id, 'access': access})
        return {**added, 'id': grant_id, 'expires': expires, 'total': len(scope['items'])}


def metadata_columns(root, columns=None):
    if columns is not None:
        if not isinstance(columns, list) or len(columns) > 30 or any(not isinstance(c, str) for c in columns):
            raise ValueError('Choose up to 30 metadata columns.')
        with database(root) as db:
            db.execute('INSERT OR REPLACE INTO preferences(key,value) VALUES(?,?)',
                       ('metadata_columns', _json(list(dict.fromkeys(columns)))))
        return columns
    if not (home(root) / 'queue.sqlite3').exists():
        return []
    with database(root) as db:
        row = db.execute("SELECT value FROM preferences WHERE key='metadata_columns'").fetchone()
        return json.loads(row['value']) if row else []


def step_state(root):
    counts = list_items(root, limit=1)['counts']
    if not counts:
        return 'none'
    return 'done' if all(r['state'] == 'applied' for r in counts) else 'partial'


def targets(root, ids):
    """Read a link's explicit scope in one query; never run a query per tile/item."""
    with database(root) as db:
        return [_item(r) for r in db.execute('SELECT * FROM items WHERE id IN (SELECT value FROM json_each(?))', (_json(ids),))]


def annotation_items(root, ids, *, image, source, polygon):
    """Find a clicked polygon only inside this link's exact source and scope."""
    with database(root) as db:
        return [_item(r) for r in db.execute(
            "SELECT * FROM items WHERE kind='annotation' "
            "AND id IN (SELECT value FROM json_each(?)) "
            "AND json_extract(target, '$.image')=? AND json_extract(target, '$.source')=? "
            "AND EXISTS (SELECT 1 FROM json_each(target, '$.ids') WHERE CAST(value AS TEXT)=?)",
            (_json(ids), str(image), str(source), str(polygon)))]


def _default(db):
    row = db.execute("SELECT g.id FROM grants g JOIN preferences p ON p.value=g.id "
                     "WHERE p.key='destination' AND g.revoked=0 AND g.expires>?", (time.time(),)).fetchone()
    return row['id'] if row else ''


def destination(root, grant_id=None):
    with database(root) as db:
        if grant_id is not None:
            if grant_id:
                row = db.execute('SELECT * FROM grants WHERE id=? AND revoked=0 AND expires>?', (grant_id,time.time())).fetchone()
                if not row:
                    raise ValueError('Choose an active reviewer link.')
            db.execute("INSERT OR REPLACE INTO preferences VALUES('destination',?)", (grant_id,))
            _event(db, None, 'owner', 'destination', {'grant':grant_id})
        return _default(db)


def _append(db, grant_id, ids):
    grant = db.execute('SELECT * FROM grants WHERE id=? AND revoked=0 AND expires>?', (grant_id,time.time())).fetchone()
    if not grant:
        raise ValueError('This reviewer link is closed.')
    if not isinstance(ids,list) or not ids:
        raise ValueError('Select items to add.')
    existing = {r[0] for r in db.execute('SELECT id FROM items WHERE id IN (SELECT value FROM json_each(?)) AND state!=?', (_json(ids),'requested'))}
    if existing != set(ids):
        raise ValueError('Approve nominations before sharing them; choose existing review items.')
    scope=json.loads(grant['scope']);before=len(scope['items'])
    scope['items']=list(dict.fromkeys([*scope['items'],*ids]))
    if len(scope['items'])>20000:
        raise ValueError('Use a separate link for more than 20,000 items.')
    db.execute('UPDATE grants SET scope=? WHERE id=?', (_json(scope),grant_id))
    if len(scope['items'])!=before:
        _event(db,None,'owner','share_add',{'grant':grant_id,'added':len(scope['items'])-before})
    return {'added':len(scope['items'])-before,'total':len(scope['items'])}


def _send_default(db, ids):
    grant_id=_default(db)
    if grant_id and ids:
        _append(db,grant_id,ids)


def append_to_grant(root, grant_id, ids):
    with database(root) as db:
        return _append(db,grant_id,ids)


def notifications(root, since=0):
    if not (home(root)/'queue.sqlite3').exists():
        return {'revision':0,'pending':0,'new':[]}
    with database(root) as db:
        events=list(db.execute("SELECT e.seq,e.action,g.name FROM events e JOIN grants g ON e.actor=g.id "
                               "WHERE e.seq>? AND e.action IN ('decision','nominate') ORDER BY e.seq LIMIT 100", (max(0,int(since)),)))
        pending=db.execute("SELECT count(*) FROM items WHERE state IN ('answered','unresolved','requested')").fetchone()[0]
        return {'revision':db.execute('SELECT coalesce(max(seq),0) FROM events').fetchone()[0],
                'pending':pending,'new':[dict(e) for e in events]}
