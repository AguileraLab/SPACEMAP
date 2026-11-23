"""Atomic copy/move of population branches, preserving their gate coordinates."""
from copy import deepcopy
import uuid

from . import flow_gating as flow
from .settings import project_lock


def transfer(channel, body):
    project, _, _ = flow.context(channel)
    with project_lock(project['root']):
        workspace = flow.load(channel)
        if body.get('revision') != workspace['revision']:
            raise ValueError('The populations changed. Try again.')
        mode, target, identifiers = body.get('mode'), body.get('target'), body.get('populations')
        by_id = {g['id']:g for g in workspace['gates']}
        if mode not in ('copy','move') or not isinstance(identifiers,list) or not identifiers:
            raise ValueError('Select populations to copy or move.')
        if any(not isinstance(i,str) or i not in by_id for i in identifiers):
            raise ValueError('Select existing populations; Analysis cells cannot be copied or moved.')
        if (target is not None and not isinstance(target,str) or target is None and mode != 'copy' or
                target is not None and target != 'root' and target not in by_id):
            raise ValueError('Choose a destination population.')
        selected = set(identifiers)
        def ancestor_selected(g):
            parent = g.get('parent','root')
            while parent != 'root':
                if parent in selected:
                    return True
                parent = by_id[parent].get('parent','root')
            return False
        roots = [g['id'] for g in workspace['gates'] if g['id'] in selected and not ancestor_selected(g)]
        included = set(roots)
        while True:
            previous = len(included)
            included.update(g['id'] for g in workspace['gates'] if g.get('parent','root') in included)
            if len(included) == previous:
                break
        if mode == 'move' and target in included:
            raise ValueError('A population cannot move into itself or its descendants.')
        if mode == 'copy' and len(by_id)+len(included) > 256:
            raise ValueError('This copy exceeds 256 populations. Remove unused branches first.')
        if target is not None and any(by_id[i].get('parent','root') != target for i in roots):
            outside = {r for i in included for r in by_id[i].get('refs',[]) if r not in included and r != 'root'}
            if outside:
                names = ', '.join(by_id[i]['name'] for i in sorted(outside))
                raise ValueError('Include the referenced populations too: '+names)
        mapping = {i:uuid.uuid4().hex for i in included} if mode == 'copy' else {i:i for i in included}
        gates = deepcopy(workspace['gates'])
        occupied = {(g.get('parent','root'),g['name']) for g in gates}
        for original in workspace['gates']:
            if original['id'] not in included:
                continue
            gate = deepcopy(original) if mode == 'copy' else next(g for g in gates if g['id']==original['id'])
            gate['id'] = mapping[original['id']]
            if original['id'] in roots:
                gate['parent'] = target if target is not None else original.get('parent','root')
                if mode == 'copy':
                    name, number = original['name'], 1
                    while (gate['parent'],name) in occupied:
                        suffix = ' copy' + (f' {number}' if number > 1 else '')
                        name = original['name'][:128-len(suffix)] + suffix
                        number += 1
                    gate['name'] = name
                    occupied.add((gate['parent'],name))
            elif mode == 'copy':
                gate['parent'] = mapping[original.get('parent','root')]
            if mode == 'copy':
                if 'refs' in gate:
                    gate['refs'] = [mapping.get(r,r) for r in gate['refs']]
                gates.append(gate)
        # Check parent/Boolean dependencies before loading any cell measurements.
        final = {g['id']:g for g in gates}
        visited, visiting = {'root'}, set()
        def visit(identifier):
            if identifier in visited:
                return
            if identifier in visiting or identifier not in final:
                raise ValueError('This move would create a population dependency cycle.')
            visiting.add(identifier)
            gate = final[identifier]
            for ref in [gate.get('parent','root'),*gate.get('refs',[])]:
                visit(ref)
            visiting.remove(identifier)
            visited.add(identifier)
        for identifier in final:
            visit(identifier)
        selected_roots = [mapping[i] for i in roots]
        if gates == workspace['gates']:
            return {'workspace':flow.public(workspace),'selected':selected_roots,'affected':0}
        # Axis fits are copied exactly. Reparenting intersects the saved gate rule
        # with its new parent's cells; it does not refit a normalization recipe.
        saved = flow.save(channel, {'revision':workspace['revision'],'changes':{'gates':gates}})
        return {**saved,'selected':selected_roots,'affected':len(included)}
