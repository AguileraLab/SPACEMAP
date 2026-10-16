/* SPACEMAP viewer, part 4 of 12: clicks on cells and labelling them.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- clicks and labelling
function cellLayer() { return state.layers.find((l) => l.type === "cells" && l.visible) || null; }

function clearCellInspection({cancelQueued=true,notify=true}={}) {
  const layer=state.inspectedCell?.layer || state.cellInspectionSelection?.layer;
  state.cardSeq+=1;state.inspectedCell=null;state.cellInspectionSelection=null;
  if(cancelQueued)state.cellInspectEpoch=(state.cellInspectEpoch||0)+1;
  $("cellcard").hidden=true;state.cellOutline=null;drawCellOutline();
  releaseSplitMarkerSelection();
  if(notify&&layer)document.dispatchEvent(new CustomEvent('spacemap-cells-picked',{detail:{layer,labels:[]}}));
}

async function onClick(event) {
  if (state.brushStroke || state.brushBusy || performance.now() < (state.brushClickUntil || 0)) return;
  if (!event.quick) return;
  if (state.pointCallback) {
    const point = toImage(event.position), done = state.pointCallback;
    state.pointCallback = null; event.preventDefaultAction = true;
    done(point ? [point.x, point.y] : null);
    return;
  }
  if (state.tool === "polygon") { event.preventDefaultAction = true; polygonClick(event.position); return; }
  if (state.tool === "merge") { event.preventDefaultAction = true; const point = toImage(event.position); if (point) mergeClick(point); return; }
  if (state.tool === "box" || state.tool === "rect" || state.tool === "freehand") return;
  const point = toImage(event.position);
  if (point && state.tool === "off") {
    state.cardSeq += 1;
    state.inspectedAnnotation = null;
    const original = event.originalEvent || {};
    // A plain click replaces the open cell at once, before anything is awaited below, so Enter or a shortcut never
    // acts on the cell shown before.
    if (!(original.ctrlKey || original.metaKey)) clearCellInspection();
    if (regionClick(point, !!(original.ctrlKey || original.metaKey || original.shiftKey))) { event.preventDefaultAction = true; return; }
    const hit = selectShapeAt(point);
    if (hit && !hit.layer.local) {
      const feature = hit.layer.features.find(f => f.properties.id === hit.id);
      showRegionCard(hit.layer, feature); event.preventDefaultAction = true; return;
    }
    if (await importedAnnotationClick(point)) {event.preventDefaultAction = true; return;}
  }
  const layer = cellLayer();
  if (!layer || !point) return;
  event.preventDefaultAction = true;
  if (state.selectionEdit) {
    try {
      const cell=await getJSON('/cells/'+layer.token+'/at?x='+point.x+'&y='+point.y+'&t='+encodeURIComponent(layer.table || '')+'&details_only=1');
      if (cell.label) state.studio?.showResult(await runCommand('refine_selection',{cells:[cell.label],mode:state.selectionEdit}));
    } catch (error) {message(error.message,true);}
    return;
  }
  if (state.tool === "click") {
    const value = event.shift ? "" : (state.labelName || "").trim();
    if (!event.shift && !value) { message("Type a name to assign first.", true); return; }
    await assign(layer, { points: [[point.x, point.y]] }, value);
    return;
  }
  const add = !!(event.originalEvent?.ctrlKey || event.originalEvent?.metaKey);
  if (!add) clearCellInspection();  // cancels a cell still opening
  if (labelSaves.queue.length || labelSaves.busy) {
    await flushLabels();
    if (labelSaves.queue.length) return;
  }
  const epoch = state.cellInspectEpoch || 0, table = layer.table;
  // Queue hits so rapid Ctrl-clicks cannot discard earlier selected cells.
  state.cellInspectQueue = (state.cellInspectQueue || Promise.resolve()).then(async () => {
    if (epoch !== (state.cellInspectEpoch || 0) || !state.layers.includes(layer) || table !== layer.table) return;
    const result = await getJSON("/cells/" + layer.token + "/at?x=" + point.x + "&y=" + point.y + "&t=" + encodeURIComponent(table || "") + "&details_only=1");
    if (epoch !== (state.cellInspectEpoch || 0) || !state.layers.includes(layer) || table !== layer.table) return;
    let selection = state.cellInspectionSelection;
    if (!add || selection?.layer !== layer || selection?.table !== table) selection = {layer, table, members:new Map()};
    state.cellInspectionSelection = selection;
    if (result.label) {
      if (add && selection.members.has(result.label)) selection.members.delete(result.label);
      else selection.members.set(result.label, {...result, point});
    }
    const seq = ++state.cardSeq, members = [...selection.members.values()];
    if (!members.length) {clearCellInspection({cancelQueued:false});return;}
    const inspected = members.length === 1 ? members[0] : selectedCellResult(members);
    document.dispatchEvent(new CustomEvent('spacemap-cells-picked', {detail: {layer, labels: members.map(m => m.label)}}));  // the Data tab follows
    state.inspectedCell = {layer, result:inspected};
    requestCellProfile(layer, inspected, seq);
    if (members.length === 1) outlineCell(layer, inspected.label, inspected.point);
    else outlineSelectedCells(layer, members, seq);
    if (layer.live && members.length === 1) showWhy(layer, inspected.point);
    if (state.spec.report_clicks) setComponentValue({event:'cell', label:result.label, labels:members.map(m=>m.label), x:point.x, y:point.y});
  }).catch(error => message(error.message, true));
}

function selectedCellResult(members) {
  const columns = new Set(members.flatMap(member => Object.keys(member.details || {}).filter(name => /^phenotype_/i.test(name))));
  const details = {};
  for (const column of columns) {
    const counts = new Map();
    members.forEach(member => {const name = String(member.details?.[column] ?? '').trim() || 'No call'; counts.set(name, (counts.get(name) || 0) + 1);});
    details[column] = [...counts].sort((a,b)=>b[1]-a[1]).map(([name,n])=>name+' ('+n+'/'+members.length+')').join(', ');
  }
  return {members, labels:members.map(member=>member.label), details};
}

async function outlineSelectedCells(layer, members, sequence) {
  state.cellOutline = null; drawCellOutline();
  let next = 0;
  await Promise.all(Array.from({length:Math.min(4,members.length)}, async () => {
    while (next < members.length && sequence === state.cardSeq) {
      const member = members[next++];
      try {
        if (!member.rings) member.rings = (await getJSON('/cells/'+layer.token+'/outline?label='+member.label+'&t='+encodeURIComponent(layer.table || ''))).rings || [];
      } catch (_) { /* Other selected outlines still identify the group. */ }
    }
  }));
  if (sequence !== state.cardSeq || !state.layers.includes(layer)) return;
  state.cellOutline = members.flatMap(member=>member.rings || []); ensureOverlay(); placeOverlay();
  scheduleSplitMarkers();
}

let dragStart = null;
function onPress(event) {
  if (state.brushBusy) { event.preventDefaultAction = true; state.brushCancelled = true; return; }
  state.brushCancelled = false;
  if (beginBrush(event)) return;
  if (state.tool === "split" && drawTarget()) {
    event.preventDefaultAction = true;
    const point = toImage(event.position);
    if (point) { drawing.start = point; drawing.points = [point, point]; }
    return;
  }
  if ((state.tool === "rect" || state.tool === "freehand") && (drawTarget() || state.selectMode && regionLayers().length)) {
    event.preventDefaultAction = true;
    const point = toImage(event.position);
    if (!point) return;
    drawing.start = point;
    drawing.points = state.tool === "freehand" ? [point] : null;
    drawing.lastScreen = event.position;
    return;
  }
  if (state.tool !== "box" || !cellLayer()) return;
  dragStart = event.position;
  event.preventDefaultAction = true;
}
function onDrag(event) {
  if (dragBrush(event)) return;
  if (drawing.start && state.tool === "split") {
    event.preventDefaultAction = true;
    const point = toImage(event.position);
    if (point) { drawing.points = [drawing.start, point]; drawPreview(); }
    return;
  }
  if (drawing.start && (state.tool === "rect" || state.tool === "freehand")) {
    event.preventDefaultAction = true;
    const point = toImage(event.position);
    if (!point) return;
    if (state.tool === "rect") {
      const s = drawing.start;
      drawing.points = [s, { x: point.x, y: s.y }, point, { x: s.x, y: point.y }];
    } else if (Math.hypot(event.position.x - drawing.lastScreen.x, event.position.y - drawing.lastScreen.y) >= 3) {
      drawing.points.push(point);
      drawing.lastScreen = event.position;
    }
    drawPreview();
    return;
  }
  if (state.tool !== "box" || !dragStart) return;
  event.preventDefaultAction = true;
  const box = $("selbox");
  const x = Math.min(dragStart.x, event.position.x), y = Math.min(dragStart.y, event.position.y);
  box.style.left = x + "px"; box.style.top = y + "px";
  box.style.width = Math.abs(event.position.x - dragStart.x) + "px";
  box.style.height = Math.abs(event.position.y - dragStart.y) + "px";
  box.hidden = false;
}
async function onRelease(event) {
  if (await endBrush(event)) return;
  if (drawing.start && state.tool === "split") {
    const [start, end] = drawing.points;
    drawing.start = null; drawing.points = null; drawPreview();
    splitAlong(start, end);
    return;
  }
  if (drawing.start && (state.tool === "rect" || state.tool === "freehand")) {
    const points = drawing.points;
    drawing.start = null; drawing.points = null;
    drawPreview();
    if (points && points.length >= 3) addShape(points);
    return;
  }
  if (state.tool !== "box" || !dragStart) return;
  const start = toImage(dragStart), end = toImage(event.position);
  dragStart = null;
  $("selbox").hidden = true;
  const layer = cellLayer();
  if (!layer || !start || !end) return;
  if (Math.abs(end.x - start.x) < 2 || Math.abs(end.y - start.y) < 2) return;
  const value = event.shift ? "" : (state.labelName || "").trim();
  if (!event.shift && !value) { message("Type a name to assign first.", true); return; }
  await assign(layer, { rect: [start.x, start.y, end.x, end.y] }, value);
}

const labelSaves = { queue: [], busy: null, error: null, timer: null, changed: new Set(), sequence: 0 };
function newSaveId() {
  const bytes = new Uint8Array(16); crypto.getRandomValues(bytes);
  return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
}
let draftOwner;
try {
  const key = 'spacemap-label-owner-' + (STANDALONE ? 'window' : 'embedded');
  draftOwner = sessionStorage.getItem(key) || newSaveId();
  sessionStorage.setItem(key, draftOwner);
} catch (_) { draftOwner = String(Date.now()) + '-' + Math.random(); }
const draftPrefix = 'spacemap-label-draft-' + draftOwner + ':';

function keepLabelDraft(item) {
  try {
    localStorage.setItem(draftPrefix + item.id, JSON.stringify({id: item.id, token: item.layer.token,
      image_token: item.layer.image_token, image_stamp: item.layer.info.fingerprint,
      table: item.table, column: item.column, value: item.value, where: item.where, frozen: item.frozen, order: item.order,
      source: item.source, provenance: item.provenance}));
  } catch (_) { message("Keep this viewer open until the labels say Saved.", true); }
}

function restoreLabelDrafts() {
  try {
    const drafts = Object.keys(localStorage).filter(k => k.startsWith(draftPrefix))
      .map(key => JSON.parse(localStorage.getItem(key))).filter(Boolean).sort((a,b) => (a.order || 0)-(b.order || 0));
    drafts.forEach(item => {
      labelSaves.sequence = Math.max(labelSaves.sequence, item.order || 0);
      if (!item || labelSaves.queue.some(q => q.id === item.id)) return;
      const layer = state.layers.find(l => l.type === 'cells' && l.token === item.token &&
        l.image_token === item.image_token && l.info.fingerprint === item.image_stamp);
      if (layer) labelSaves.queue.push({...item, layer});
    });
    if (labelSaves.queue.length) {
      message("Restoring labels that were waiting to save.");
      flushLabels();
    }
  } catch (_) { /* a private browser may disable local storage */ }
}

function updateLabelStatus() {
  const count = labelSaves.queue.reduce((n, item) => n + (item.where.points?.length || 1), 0);
  const kind = labelSaves.error ? "error" : labelSaves.busy ? "busy" : count ? "waiting" : "saved";
  const text = {error: "Not saved", busy: "Saving...", waiting: count + " waiting", saved: "Saved"}[kind];
  const tip = {error: labelSaves.error, busy: "Saving in the background.", waiting: count + " change(s) waiting to save.",
    saved: "Every label is saved."}[kind];
  document.querySelectorAll('.label-save-status').forEach(box => {
    box.textContent = text; box.title = tip;
    box.className = 'chip label-save-status ' + ({error: 'error', busy: 'warn', waiting: 'warn', saved: 'ok'}[kind]);
  });
  document.querySelectorAll('.label-save-error').forEach(box => { box.hidden = !labelSaves.error; box.textContent = labelSaves.error || ''; });
  // Save and discard only appear while something waits, so a saved panel stays quiet.
  document.querySelectorAll('.label-save-now, .label-discard').forEach(button => {
    button.hidden = !count; button.disabled = !!labelSaves.busy || !count;
  });
  drawLabelPreview();
}

function assign(layer, where, value, source = 'typed', provenance = null) {
  const column = (state.labelColumn || "").trim();
  if (!column) { message("Choose the column to write labels into.", true); return; }
  let item = labelSaves.queue[labelSaves.queue.length - 1];
  if (item && !item.frozen && item.layer === layer && item.table === layer.table && item.column === column &&
      item.value === value && (item.source || 'typed') === source && !provenance && item.where.points && where.points && item.where.points.length < 100) {
    item.where.points.push(...where.points);
  } else {
    item = {id: newSaveId(), order: ++labelSaves.sequence, layer, table: layer.table, column, value, where, frozen: false, source, provenance};
    labelSaves.queue.push(item);
  }
  keepLabelDraft(item);
  updateLabelStatus();
  // A fixed short window batches rapid clicks without postponing saving
  // indefinitely while the user continues to click.
  if (!labelSaves.timer && !labelSaves.busy && !labelSaves.error) {
    labelSaves.timer = setTimeout(() => { labelSaves.timer = null; flushLabels(); }, 500);
  }
}

async function flushLabels() {
  clearTimeout(labelSaves.timer); labelSaves.timer = null;
  if (labelSaves.busy) return labelSaves.busy;
  labelSaves.error = null;
  const run = async () => {
    if (labelSaves.queue.length) {
      try {
        const service = await getJSON('/health');
        if (!service.label_batches) throw new Error('Restart SPACEMAP, then reopen this viewer to use the new saving controls.');
      } catch (error) { labelSaves.error = error.message; return; }
    }
    while (labelSaves.queue.length) {
      const item = labelSaves.queue[0];
      item.frozen = true; keepLabelDraft(item);
      try {
        const result = await postJSON('/cells/' + item.layer.token + '/assign', {
          t: item.table, column: item.column, value: item.value, ...item.where, request_id: item.id,
          source: item.where.values ? 'undo' : (item.source || 'typed'), provenance: item.provenance});
        if (!Array.isArray(result.labels)) throw new Error('The viewer service changed. Restart SPACEMAP and retry Save now.');
        labelSaves.queue.shift();
        try { localStorage.removeItem(draftPrefix + item.id); } catch (_) { /* keep the receipt safe to retry */ }
        if (result.assigned) {
          if (!item.where.values) state.undo.push({layer: item.layer, table: item.table, column: item.column, previous: result.previous,
            expected: Object.fromEntries(result.labels.map(label => [String(label), item.value]))});
          const layer = item.layer;
          if (layer.table === item.table) {
            if (layer.columns && !layer.columns.includes(item.column)) layer.columns.push(item.column);
            layer.label_columns = Array.from(new Set((layer.label_columns || []).concat([item.column])));
          }
          if (state.layers.includes(layer) && layer.table === item.table && state.labelColumn === item.column) {
            if (layer.live) layer.referencePhenotypes = layer.live.phenotypes.slice();
            layer.live = null;
            layer.color_by = item.column;
            if (layer.mode === 'outline') layer.mode = 'both';
          }
          labelSaves.changed.add(layer);
          document.dispatchEvent(new CustomEvent('spacemap-reference-updated',{detail:{column:item.column}}));
        } else message('No numbered cell was found at that spot.');
      } catch (error) {
        labelSaves.error = error.message;
        message('Labels are waiting to save. Use Save now to retry.', true);
        break;
      }
    }
  };
  labelSaves.busy = run();
  updateLabelStatus();
  await labelSaves.busy;
  labelSaves.busy = null;
  updateLabelStatus();
  if (!labelSaves.error) debounce('label-colours', refreshLabelColours, 700);
}

function refreshLabelColours() {
  // Repaint after a pause, without rebuilding inputs under the user's cursor.
  if (labelSaves.queue.length || labelSaves.busy) return;
  labelSaves.changed.forEach(layer => {
    if (!state.layers.includes(layer)) return;
    state.versions[layer.id] = (state.versions[layer.id] || 0) + 1;
    refreshLayer(layer);
    if (layer.refreshCellControls) layer.refreshCellControls();
    if (layer.refreshReferenceCounts) layer.refreshReferenceCounts();
    if (layer.refreshReview) layer.refreshReview();
  });
  labelSaves.changed.clear(); saveView();
}

function drawLabelPreview() {
  if (!state.svgRoot) return;
  if (state.labelPreview) state.labelPreview.remove();
  const group = document.createElementNS(SVG_NS, 'g');
  group.setAttribute('class', 'label-preview');
  labelSaves.queue.filter(item => state.layers.includes(item.layer)).forEach(item => {
    (item.where.points || []).forEach(([x,y]) => {
      const dot = document.createElementNS(SVG_NS, 'circle');
      dot.setAttribute('cx', x); dot.setAttribute('cy', y);
      dot.setAttribute('r', 7 / Math.max(state.overlayScale || 1, 1e-6));
      group.appendChild(dot);
    });
    if (item.where.rect) {
      const [x0,y0,x1,y1] = item.where.rect;
      const rect = document.createElementNS(SVG_NS, 'rect');
      Object.entries({x: Math.min(x0,x1), y: Math.min(y0,y1), width: Math.abs(x1-x0), height: Math.abs(y1-y0)})
        .forEach(([key,value]) => rect.setAttribute(key,value));
      group.appendChild(rect);
    }
  });
  state.svgRoot.appendChild(group); state.labelPreview = group;
}

async function undo() {
  const queued = labelSaves.queue[labelSaves.queue.length-1];
  if (queued && !queued.frozen) {
    if (queued.where.points?.length > 1) { queued.where.points.pop(); keepLabelDraft(queued); }
    else { labelSaves.queue.pop(); try { localStorage.removeItem(draftPrefix + queued.id); } catch (_) {} }
    if (!labelSaves.queue.length) labelSaves.error = null;
    updateLabelStatus(); message('Last waiting change removed.'); return;
  }
  await flushLabels();
  if (labelSaves.queue.length) return;
  const last = state.undo.pop();
  if (!last) { message("Nothing to undo."); return; }
  const item = {id: newSaveId(), order: ++labelSaves.sequence, layer: last.layer, table: last.table, column: last.column,
    value: '', where: {values: last.previous, expected: last.expected}, frozen: false};
  labelSaves.queue.push(item); keepLabelDraft(item); await flushLabels();
}

window.addEventListener('beforeunload', event => {
  if (labelSaves.queue.length || labelSaves.busy) { event.preventDefault(); event.returnValue = ''; }
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && labelSaves.queue.length && !labelSaves.error) flushLabels();
});

async function inspectReviewedCell(layer, label) {
  const asked = state.cardSeq;
  await flushLabels();
  if (labelSaves.queue.length || asked !== state.cardSeq) return;  // a click on another cell meanwhile wins
  state.cellInspectEpoch = (state.cellInspectEpoch || 0) + 1;
  const seq = ++state.cardSeq;
  const result = await getJSON('/cells/' + layer.token + '/at?label=' + label + '&details_only=1&t=' + encodeURIComponent(layer.table));
  if (seq !== state.cardSeq || !state.layers.includes(layer)) return;
  state.inspectedCell = {layer, result};
  state.cellInspectionSelection = {layer, table:layer.table, members:new Map([[label,result]])};
  const x = Number(result.details?.x_centroid), y = Number(result.details?.y_centroid);
  outlineCell(layer, label, Number.isFinite(x) && Number.isFinite(y) ? {x, y} : null);
  const item = baseItem();
  if (Number.isFinite(x) && Number.isFinite(y) && item) {
    state.viewer.viewport.fitBounds(item.imageToViewportRectangle(x-100, y-100, 200, 200));
  }
  requestCellProfile(layer, result, seq);
}

function cellReviewControl(layer, result, card) {
  if (result.members) return;
  if (!layer.table || !state.labelColumn) return;
  const column = state.labelColumn, key = layer.table + '/' + column;
  if (result.reviewControl?.key === key) { card.appendChild(result.reviewControl.node); return; }
  const button = el('button', {class: 'btn small', text: 'Mark needs review', disabled: true});
  const checked = el('button',{class:'btn small',text:'Mark checked',disabled:true});
  const row = el('div', {class: 'row'}, [button,checked]);
  result.reviewControl = {key, node: row};
  const setText = () => {
    button.textContent = result.reviewFlag ? 'Needs review · mark checked' : 'Mark needs review';
    button.disabled = false;
    button.setAttribute('aria-pressed', String(!!result.reviewFlag));
    checked.textContent=result.reviewChecked?'Checked':'Mark checked';checked.disabled=!!result.reviewChecked;
  };
  const route = '/cells/' + layer.token + '/review';
  getJSON(route + '?t=' + encodeURIComponent(layer.table) + '&column=' + encodeURIComponent(column) + '&label=' + result.label)
    .then(value => { result.reviewFlag = value.flagged; result.reviewChecked=value.checked;setText(); })
    .catch(error => { button.textContent = 'Review status unavailable'; button.title = error.message; });
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      await flushLabels();
      if (labelSaves.queue.length) { setText(); return; }
      const value = await postJSON(route, {t: layer.table, column, label: result.label, flagged: !result.reviewFlag});
      result.reviewFlag = value.flagged;result.reviewChecked=value.checked; setText();
      if (layer.refreshReview) layer.refreshReview();
    } catch (error) { setText(); message(error.message, true); }
  });
  checked.addEventListener('click',async()=>{
    checked.disabled=true;
    try{
      await flushLabels();if(labelSaves.queue.length){setText();return;}
      const value=await postJSON(route,{t:layer.table,column,label:result.label,flagged:false});
      result.reviewFlag=value.flagged;result.reviewChecked=value.checked;setText();layer.refreshReview?.();
    }catch(error){setText();message(error.message,true);}
  });
  card.appendChild(row);
}

function bump(layer) {
  state.versions[layer.id] = (state.versions[layer.id] || 0) + 1;
  refreshLayer(layer);
  buildPanel();
  saveView();
}

function showCellCard(layer, result) {
  splitMarkerSelection(layer, result);
  const card = $("cellcard"), scroll = card.scrollTop, focus = document.activeElement;
  fillCellCard(layer, result);
  if (focus?.isConnected && card.contains(focus)) focus.focus({preventScroll: true});
  card.scrollTop = scroll;
  document.querySelectorAll('.cell-marker-controls').forEach(box=>box.refresh?.());
  syncMarkerTransformControls(layer);
}

document.addEventListener('spacemap-reference-updated',()=>debounce('cell-details-after-labels',async()=>{
  const inspected=state.inspectedCell;if(!inspected)return;
  const {layer,result}=inspected, table=layer.table, members=result.members || [result];
  const current=()=>state.inspectedCell===inspected&&state.layers.includes(layer)&&layer.table===table;
  let next=0;
  try{
    await Promise.all(Array.from({length:Math.min(4,members.length)},async()=>{
      while(next<members.length&&current()){
        const member=members[next++];
        const fresh=await getJSON('/cells/'+layer.token+'/at?details_only=1&label='+member.label+'&t='+encodeURIComponent(table));
        if(current()&&fresh.details)member.details=fresh.details;
      }
    }));
    if(!current())return;
    if(result.members)result.details=selectedCellResult(members).details;
    showCellCard(layer,result);
  }catch(error){if(current())message('Labels were saved, but the cell card could not refresh: '+error.message,true);}
},200));

function fillCellCard(layer, result) {
  const card = $("cellcard");
  card.innerHTML = "";
  const close = el("button", { class: "btn small", text: "Close", onclick: () => {
    clearCellInspection();
  } });
  card.appendChild(el("h4", {}, [el("span", { text: result.members ? result.members.length + ' cells selected' : "Cell " + result.label }), close]));
  if (result.members) card.appendChild(el('div', {class:'hint', text:'Ctrl-click to add or remove cells. Click normally for one cell.'}));
  const details = result.details || {};
  if (details._note) card.appendChild(el("div", { class: "hint", text: details._note }));
  phenotypeCalls(layer, result, details, card);
  if (result.why) renderWhy(result.why, card);
  renderSuggestions(layer, result, card);
  cellReviewControl(layer, result, card);
  // Marker bars, folded; open or closed, the choice carries over to the next cell.
  const markers = el("details", { class: "cell-markers" });
  markers.open = !!state.markersOpen;
  markers.appendChild(el("summary", { text: "Markers" }));
  markers.addEventListener("toggle", () => { if (markers.isConnected) state.markersOpen = markers.open; });
  if (result.profile) renderCellProfile(layer, result.profile, markers);
  else rankedMarkers(layer, result, details, markers);
  card.appendChild(markers);
  const channels = new Set(((state.base && state.base.info.channels) || []).map(String));
  const labelLike = /cluster|neigh|^cn|zone|bed|tissue|roi$|class|type/i;
  const others = Object.entries(details).filter(([key, value]) => !key.startsWith("_") && !/^phenotype_/i.test(key) &&
    !channels.has(key) && typeof value === "string" && value !== "" && labelLike.test(key));
  if (others.length) {
    const folded = el("details", { class: "cell-other-labels" });
    folded.appendChild(el("summary", { text: "Other labels (" + others.length + ")" }));
    folded.appendChild(el("table", { class: "calls" }, others.map(([k, v]) => el("tr", {}, [el("td", { class: "k", text: k }), el("td", { text: v })]))));
    card.appendChild(folded);
  }
  card.hidden = false;
  document.querySelector('#ctxmenu .cell-phenotype-menu')?.refresh?.();
}

const MAIN_PHENOTYPE_KEY = "spacemap-viewer-main-phenotype";

// The cell's call in every phenotype column; the main column (chosen, remembered) first.
function phenotypeCalls(layer, result, details, card) {
  const columns = Object.keys(details).filter((key) => /^phenotype_/i.test(key)).sort((a, b) => a.localeCompare(b));
  if (!columns.length) return;
  let saved = null;
  try { saved = localStorage.getItem(MAIN_PHENOTYPE_KEY); } catch (_) { /* private window */ }
  let main = state.mainPhenotype || saved;
  if (!columns.includes(main)) main = columns.includes("phenotype_ml_consensus") ? "phenotype_ml_consensus" : columns[0];
  const short = (column) => column.replace(/^phenotype_/i, "");
  const call = (column) => {
    const value = details[column];
    return value === null || value === undefined || String(value).trim() === "" ? null : String(value);
  };
  const picker = select(columns.map((c) => [c, short(c)]), main, (value) => {
    state.mainPhenotype = value;
    try { localStorage.setItem(MAIN_PHENOTYPE_KEY, value); } catch (_) { /* private window */ }
    showCellCard(layer, result);
  });
  picker.setAttribute("aria-label", "Main phenotype column");
  const mainCall = call(main);
  const box = el("div", { class: "cell-calls" });
  card.appendChild(box);
  box.appendChild(el("div", { class: "row call-main" }, [chip(mainCall || "No call", mainCall ? "main" : "", main), picker,
    help("The cell's phenotype in the main column, chosen here and remembered, then its call in every other " +
      "phenotype column. Calls that differ from the main one are highlighted. ref columns hold manual labels, " +
      "including labels saved but not yet written into the CSV.")]));
  const rows = columns.filter((column) => column !== main).map((column) => {
    const value = call(column);
    const differs = value !== null && mainCall !== null && value !== mainCall;
    return el("tr", { class: differs ? "differs" : value === null ? "none" : "" },
      [el("td", { class: "k", text: short(column) }), el("td", { text: value === null ? "none" : value })]);
  });
  if (rows.length) box.appendChild(el("table", { class: "calls" }, rows));
}

// Marker bars from the ranking returned with the click (when no profile was requested).
function rankedMarkers(layer, result, details, box) {
  const ranking = result.ranking || [];
  const channels = new Set(((state.base && state.base.info.channels) || []).map(String));
  if (ranking.length) {
    const choices = [["z1", "Z1", "How unusually high this cell is for each marker, compared with the other cells of this ROI (as RESOLVE's Z1, Arcsinh first)"],
      ["z2", "Z2", "Which markers stand out within this cell: its Z1 values as a z-score across its own markers (RESOLVE's Z2)"],
      ["asinh", "Arcsinh", "arcsinh(raw / 5)"], ["raw", "Raw", "Measured mean intensity"]];
    const by = state.rankBy || "z1";
    const seg = el("div", { class: "seg rank" });
    choices.forEach(([key, text, tip]) => seg.appendChild(el("button", { type: "button", text: text, title: tip, class: key === by ? "on" : "",
      onclick: () => { state.rankBy = key; showCellCard(layer, result); } })));
    box.appendChild(seg);
    const sorted = ranking.slice().sort((a, b) => b[by] - a[by]).slice(0, 15);
    const top = Math.max(...sorted.map((m) => Math.abs(m[by])), 1e-9);
    box.appendChild(el("table", {}, sorted.map((m) => el("tr", { title: "Raw " + number(m.raw) + ", Arcsinh " + m.asinh.toFixed(2) +
        ", Z1 " + m.z1.toFixed(2) + ", Z2 " + m.z2.toFixed(2) }, [
      el("td", { class: "k", text: m.marker }),
      el("td", {}, [el("div", { text: (by === "raw" ? number(m[by]) : m[by].toFixed(2)) }),
        el("div", { class: "bar" + (m[by] < 0 ? " neg" : ""), style: "width:" + Math.max(2, 100 * Math.abs(m[by]) / top) + "%" })]),
    ]))));
    return;
  }
  const markers = Object.entries(details).filter(([key, value]) => channels.has(key) && typeof value === "number")
    .sort((a, b) => b[1] - a[1]).slice(0, 15);
  if (!markers.length) { box.appendChild(el("div", { class: "hint", text: "No marker values for this cell." })); return; }
  const max = Math.max(...markers.map((m) => m[1]), 1e-9);
  box.appendChild(el("table", {}, markers.map(([k, v]) => el("tr", {}, [
    el("td", { class: "k", text: k }),
    el("td", {}, [el("div", { text: v.toFixed(1) }), el("div", { class: "bar", style: "width:" + Math.max(2, 100 * v / max) + "%" })]),
  ]))));
}

function setProfileSource(layer, value) {
  layer.profileSource = value;
  if (value !== 'table') layer.markerTransforms = (layer.markerTransforms || []).filter(step => step !== 'Z1');
  document.querySelectorAll('.profile-source').forEach(input => { input.value = value; });
  saveView();
  if (state.inspectedCell?.layer === layer) requestCellProfile(layer, state.inspectedCell.result, ++state.cardSeq);
}

function profileChoice(layer) {
  const options = [["table", "Saved measurements"], ["image", "Image pixels"]];
  if (layer.has_background_image) options.push(["corrected", "Background removed (Tab 2)"]);
  const choice = select(options, layer.profileSource || "table", value => setProfileSource(layer, value));
  choice.classList.add("profile-source");
  choice.setAttribute("aria-label", "Cell measurements from");
  return choice;
}

async function requestCellProfile(layer, result, sequence, preserveSuggestions = false) {
  if (result.members) return requestSelectedCellProfiles(layer, result, sequence);
  const keepSuggestions = preserveSuggestions && result.suggestions?.status === 'ready' &&
    result.suggestionSettings === JSON.stringify(suggestionSettings(layer));
  if (!keepSuggestions) { delete result.suggestions; delete result.acceptedSuggestion; }
  result.profile = { status: "preparing", message: "Reading this cell's markers..." };
  showCellCard(layer, result);
  $("cellcard").scrollTop = 0;
  if (!keepSuggestions && layer.suggestionsOn && layer.suggestionMode !== 'examples') requestSuggestions(layer, result, sequence);
  const image = state.layers.find((l) => l.type === "image" && l.token === layer.image_token);
  const backgrounds = {};
  ((image && image.channels) || []).forEach((c) => { if (Number.isFinite(c.low)) backgrounds[c.name] = c.low; });
  const payload = { label: result.label, source: layer.profileSource || "table", t: layer.table || "", backgrounds, transforms: layer.markerTransforms || [], retry: true };
  try {
    while (sequence === state.cardSeq && state.layers.includes(layer)) {
      const profile = await postJSON("/cells/" + layer.token + "/profile", payload);
      payload.retry = false;
      if (sequence !== state.cardSeq || !state.layers.includes(layer)) return;
      result.profile = profile;
      if (profile.status === 'ready') result.profileKey = JSON.stringify({...payload, retry:undefined, label:undefined});
      showCellCard(layer, result);
      if (profile.status === 'ready') applyCellMarkers(state.inspectedCell);
      if (profile.status !== "preparing") {
        if (!keepSuggestions && profile.status === 'ready' && layer.suggestionsOn && layer.suggestionMode === 'examples') requestSuggestions(layer, result, sequence);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  } catch (error) {
    if (sequence !== state.cardSeq) return;
    result.profile = { status: "error", message: error.message };
    showCellCard(layer, result);
  }
}

function averageCellProfiles(profiles) {
  const collected = new Map();
  profiles.forEach(profile => (profile.markers || []).forEach(row => {
    if (!collected.has(row.marker)) collected.set(row.marker, []);
    collected.get(row.marker).push(row);
  }));
  const markers = [...collected].map(([marker, rows]) => {
    const result = {marker, cells:rows.length};
    for (const key of ['mean','signal','view','background','positive_percent']) {
      const values = rows.map(row=>row[key]).filter(Number.isFinite);
      result[key] = values.length ? values.reduce((sum,v)=>sum+v,0) / values.length : null;
    }
    return result;
  });
  return {...profiles[0], label:undefined, markers, count:profiles.length,
    message:'Average across '+profiles.length+' cells; transformations are applied before averaging.',
    missing_backgrounds:[...new Set(profiles.flatMap(p=>p.missing_backgrounds || []))]};
}

async function requestSelectedCellProfiles(layer, result, sequence) {
  result.profile = {status:'preparing', message:'Reading '+result.members.length+' selected cells…'};
  delete result.suggestions; delete result.acceptedSuggestion;
  showCellCard(layer, result);
  const image = state.layers.find(l=>l.type==='image'&&l.token===layer.image_token), backgrounds = {};
  (image?.channels || []).forEach(c=>{if(Number.isFinite(c.low)) backgrounds[c.name]=c.low;});
  const payload = {source:layer.profileSource || 'table', t:layer.table || '', backgrounds, transforms:layer.markerTransforms || []};
  const key = JSON.stringify(payload), profiles = new Array(result.members.length);
  const current = () => sequence===state.cardSeq && state.layers.includes(layer) && state.inspectedCell?.result===result;
  let next=0;
  try {
    await Promise.all(Array.from({length:Math.min(4,result.members.length)}, async()=>{
      while(next<result.members.length && current()){
        const index=next++, member=result.members[index];
        if(member.profileKey===key && member.profile?.status==='ready'){profiles[index]=member.profile;continue;}
        let retry=true;
        while(current()){
          const profile=await postJSON('/cells/'+layer.token+'/profile',{...payload,label:member.label,retry});retry=false;
          if(!current())return;
          if(profile.status==='ready'){member.profile=profile;member.profileKey=key;profiles[index]=profile;break;}
          if(profile.status!=='preparing')throw Error('Cell '+member.label+': '+(profile.message || 'Measurements unavailable.'));
          await new Promise(resolve=>setTimeout(resolve,500));
        }
      }
    }));
    if(!current())return;
    result.profile=averageCellProfiles(profiles);showCellCard(layer,result);applyCellMarkers(state.inspectedCell);
    if(layer.suggestionsOn)requestSuggestions(layer,result,sequence);
  }catch(error){if(current()){result.profile={status:'error',message:error.message};showCellCard(layer,result);}}
}

function suggestionSettings(layer) {
  const destination=state.labelColumn || 'phenotype_ref1';
  if (layer.suggestionMode === 'saved_reference') return {mode:'saved_reference',model_id:layer.suggestionModelId,t:layer.table,destination};
  if ((layer.suggestionMode || 'consensus_rf') === 'consensus_rf') {
    return {mode: 'consensus_rf', t: layer.table, destination};
  }
  const image = state.layers.find(l => l.type === 'image' && l.token === layer.image_token);
  const backgrounds = {};
  (image?.channels || []).forEach(c => { if (Number.isFinite(c.low)) backgrounds[c.name] = c.low; });
  return {t: layer.table, source: layer.profileSource || 'table', backgrounds,
    column: layer.learnColumn || destination, destination};
}

function referencePreferenceKey(table) {
  return 'spacemap-reference-choice:'+String(state.spec?.workspace || '')+':'+table;
}

function rememberReferenceChoice(layer) {
  const table=layer.table===layer.combinedTable ? dataView.combined?.path : layer.referenceModelTable;
  if(!table)return;
  try{localStorage.setItem(referencePreferenceKey(table),JSON.stringify({mode:layer.suggestionMode,
    model:layer.suggestionModelId,on:!!layer.suggestionsOn,column:layer.learnColumn}));}catch(_){/* private mode */}
}

function restoreReferenceChoice(layer, table) {
  if(layer.referenceChoiceTable===table)return;
  layer.referenceChoiceTable=table;layer.referenceModelTable=table;
  let chosen;
  try{chosen=JSON.parse(localStorage.getItem(referencePreferenceKey(table)) || 'null');}catch(_){/* private mode */}
  if(chosen){
    layer.suggestionMode=chosen.mode;layer.suggestionModelId=chosen.model;layer.suggestionsOn=chosen.on;
    if(chosen.column)layer.learnColumn=chosen.column;
  }else if(state.studio){
    // Per-image views from older Studio versions defaulted to consensus_rf.
    // A new ROI should look for the shared reference model, not require an
    // unrelated phenotype_ml_consensus result column.
    layer.suggestionMode='saved_reference';layer.suggestionModelId=null;
  }
}

async function requestSuggestions(layer, result, sequence) {
  if(layer.table===layer.combinedTable&&dataView.combined?.path)restoreReferenceChoice(layer,dataView.combined.path);
  if(state.studio&&!layer.suggestionMode)layer.suggestionMode='saved_reference';
  if(layer.suggestionMode==='saved_reference'&&!layer.suggestionModelId){
    result.suggestions={status:'preparing',message:'Finding a saved reference model…'};showCellCard(layer,result);
    try{
      const found=await getJSON('/cells/'+layer.token+'/reference_models?t='+encodeURIComponent(layer.table || ''));
      if(sequence!==state.cardSeq||!state.layers.includes(layer)||!layer.suggestionsOn)return;
      layer.referenceModelTable=found.table;
      if(!useReferenceModel(layer,found.models))throw Error('No compatible reference model yet. Train from this reference, or choose a model from the library.');
      rememberReferenceChoice(layer);
    }catch(error){if(sequence===state.cardSeq){result.suggestions={status:'error',message:error.message};showCellCard(layer,result);}return;}
  }
  if (result.members) return requestSelectedSuggestions(layer, result, sequence);
  const settings = suggestionSettings(layer), signature = JSON.stringify(settings);
  result.suggestionSettings = signature;
  result.suggestions = {status: 'preparing', message: 'Preparing suggestions...'};
  showCellCard(layer, result);
  let retry = true;
  try {
    while (sequence === state.cardSeq && state.layers.includes(layer) && layer.suggestionsOn) {
      const value = await postJSON('/cells/' + layer.token + '/suggestions', {...settings, label: result.label, retry});
      retry = false;
      if (sequence !== state.cardSeq || !layer.suggestionsOn) return;
      if (signature !== JSON.stringify(suggestionSettings(layer))) {
        // Combined-table linking and the saved-model choice can finish just
        // after the first click on a new image. Discard the old response and
        // use the current inputs without making the user click again.
        if(state.inspectedCell?.result===result&&state.layers.includes(layer))return requestSuggestions(layer,result,sequence);
        return;
      }
      result.suggestions = value; showCellCard(layer, result);
      if (value.status !== 'preparing') return;
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  } catch (error) {
    if (sequence !== state.cardSeq) return;
    if(layer.suggestionsOn&&state.inspectedCell?.result===result&&state.layers.includes(layer)&&
       signature!==JSON.stringify(suggestionSettings(layer)))return requestSuggestions(layer,result,sequence);
    result.suggestions = {status: 'error', message: error.message === 'HTTP 404' ?
      'Suggestions could not be loaded. Retry below. If this continues, restart SPACEMAP and reopen the viewer.' : error.message};
    showCellCard(layer, result);
  }
}

function useReferenceModel(layer, models, preferred) {
  const available=models.filter(model=>model.model_id);
  if(!available.length){layer.suggestionModelId=null;return false;}
  const find=id=>available.find(model=>model.model_id===id) || available.find(model=>model.replaces?.includes(id));
  const selected=find(preferred) || find(layer.suggestionModelId);
  if(selected)layer.suggestionModelId=selected.model_id;
  else if(!available.some(model=>model.model_id===layer.suggestionModelId)){
    const related=available.filter(model=>model.column===(layer.learnColumn || state.labelColumn));
    layer.suggestionModelId=(related[0] || available[0]).model_id;
  }
  saveView();return true;
}

const appliedAutoModels=new WeakMap();
document.addEventListener('spacemap-auto-reference',event=>{
  const value=event.detail || {};
  state.autoReference=value;
  document.querySelectorAll('.manual-reference-card').forEach(card=>card.applyAuto?.(value));
  if(value.enabled&&value.completed?.result?.models?.length&&state.layers.some(layer=>
      layer.type==='cells'&&layer.table&&appliedAutoModels.get(layer)!==value.completed.id)){
    document.dispatchEvent(new CustomEvent('spacemap-reference-trained',{detail:value.completed}));
  }
});

document.addEventListener('spacemap-reference-trained',async event=>{
  const job=event.detail, result=job?.result;
  if(!result?.models?.length)return;
  const best=result.models.slice().sort((a,b)=>(b.held_out_balanced_accuracy ?? -1)-(a.held_out_balanced_accuracy ?? -1))[0];
  const version=job.id || best.model_id;
  for(const layer of state.layers.filter(item=>item.type==='cells'&&item.table)){
    if(appliedAutoModels.get(layer)===version)continue;
    appliedAutoModels.set(layer,version);
    const table=layer.table;
    try{
      const found=await getJSON('/cells/'+layer.token+'/reference_models?t='+encodeURIComponent(table));
      if(!state.layers.includes(layer)||layer.table!==table||found.table!==result.table){appliedAutoModels.delete(layer);continue;}
      if(!useReferenceModel(layer,found.models,best.model_id))continue;
      layer.referenceModelTable=found.table;layer.referenceChoiceTable=found.table;
      layer.suggestionMode='saved_reference';layer.suggestionsOn=true;rememberReferenceChoice(layer);saveView();
      document.querySelectorAll('.manual-reference-card').forEach(card=>card.refreshModels?.());
      if(state.inspectedCell?.layer===layer)requestSuggestions(layer,state.inspectedCell.result,state.cardSeq);
    }catch(error){appliedAutoModels.delete(layer);message('Reference training finished, but suggestions could not refresh: '+error.message,true);}
  }
});

async function requestSelectedSuggestions(layer, result, sequence) {
  const settings=suggestionSettings(layer), signature=JSON.stringify(settings);
  result.suggestionSettings=signature;
  result.suggestions={status:'preparing',message:'Combining predictions for '+result.members.length+' cells…'};
  showCellCard(layer,result);
  const current=()=>sequence===state.cardSeq && state.layers.includes(layer) && layer.suggestionsOn &&
    state.inspectedCell?.result===result && signature===JSON.stringify(suggestionSettings(layer));
  const values=new Array(result.members.length);let next=0;
  await Promise.all(Array.from({length:Math.min(4,result.members.length)},async()=>{
    while(next<result.members.length && current()){
      const index=next++, member=result.members[index];let retry=true;
      try{
        while(current()){
          const value=await postJSON('/cells/'+layer.token+'/suggestions',{...settings,label:member.label,retry});retry=false;
          if(!current())return;
          if(value.status!=='preparing'){values[index]=value;break;}
          await new Promise(resolve=>setTimeout(resolve,500));
        }
      }catch(error){values[index]={status:'error',message:error.message};}
    }
  }));
  if(!current())return;
  const ready=values.filter(v=>v?.status==='ready'), sums=new Map();
  if(!ready.length){result.suggestions={status:'error',message:values.find(v=>v?.message)?.message || 'No predictions available.'};showCellCard(layer,result);return;}
  ready.forEach(value=>(value.probabilities || value.choices).forEach(choice=>sums.set(choice.phenotype,(sums.get(choice.phenotype)||0)+choice.probability)));
  result.suggestions={status:'ready',provenance:{...ready[0].provenance,
    missing_markers:[...new Set(ready.flatMap(v=>v.provenance?.missing_markers || []))]},scored:ready.length,total:result.members.length,
    choices:[...sums].map(([phenotype,sum])=>({phenotype,probability:sum/ready.length})).sort((a,b)=>b.probability-a.probability).slice(0,3)};
  showCellCard(layer,result);
}

function acceptSuggestion(layer, result, choice) {
  if (!layer.suggestionsOn || result.acceptedSuggestion || state.inspectedCell?.result !== result) return;
  if (result.suggestionSettings !== JSON.stringify(suggestionSettings(layer))) {
    message('The measurement settings changed. Click the cell again before accepting.', true); return;
  }
  assign(layer, {labels: result.labels || [result.label]}, choice.phenotype, 'accepted',
    {...result.suggestions.provenance, probability: choice.probability, phenotype: choice.phenotype});
  result.acceptedSuggestion = choice.phenotype;
  showCellCard(layer, result);
}

function renderSuggestions(layer, result, card) {
  const box = el('div', {class: 'cell-suggestions'});
  const ai = state.studio ? el('button', {class:'btn small cell-ask-ai', text:result.aiBusy?'Preparing…':'Ask AI',
    disabled:result.aiBusy || result.profile?.status!=='ready' ? '' : null,
    title:'Ask the Assistant using this cell’s measurements, morphology, marker tiles and existing phenotype names. Labels stay unchanged.',
    onclick:()=>askCellAI(layer,result,ai)}) : null;
  if(ai)box.appendChild(ai);
  box.appendChild(el('div', {class: 'section', text: 'Suggested phenotypes'}));
  if (!layer.suggestionsOn) {
    box.appendChild(el('button', {class: 'btn small', text: 'Show suggestions', onclick: () => {
      layer.suggestionsOn = true;
      document.querySelectorAll('[aria-label="Suggest phenotypes"]').forEach(input => { input.checked = true; });
      saveView(); requestCellProfile(layer, result, ++state.cardSeq);
    }}));
    card.appendChild(box); return;
  }
  const value = result.suggestions;
  if (!value || value.status !== 'ready') {
    const examples = layer.suggestionMode === 'examples';
    const missing = examples && result.profile?.missing_backgrounds?.length;
    box.appendChild(el('div', {class: 'hint', role: 'status', text: missing && layer.profileSource === 'image' ?
      'Set backgrounds for ' + missing + (missing === 1 ? ' marker' : ' markers') + ' with Auto all markers, or use saved measurements for suggestions.' :
      value?.message || (examples ? result.profile?.message : '') || 'Preparing the saved model...'}));
    if (examples && (layer.profileSource || 'table') !== 'table') box.appendChild(el('button', {class: 'btn small', text: 'Use saved measurements',
      onclick: () => setProfileSource(layer, 'table')}));
    if (value?.status === 'error') box.appendChild(el('button', {class: 'btn small', text: 'Retry suggestions',
      onclick: () => requestCellProfile(layer, result, ++state.cardSeq)}));
  } else {
    const p = value.provenance;
    const source = {table: 'saved measurements', image: 'image pixels', corrected: 'image pixels with background removed'}[p.source];
    const notes = [result.members ? 'Average model support across selected cells, not certainty. Accepting assigns this label to every selected cell.' : 'Estimated matches, not certainty; nothing is assigned until you accept. Enter accepts the first.',
      'Learned from ' + Number(p.examples).toLocaleString() + ' labelled cells in ' + p.column + ' (' + p.roi + ', ' + p.table + '), using ' + source + '.',
      'Cells flagged for review in that column are left out.'];
    if (p.method === 'Random Forest' && !p.model_id) notes.push('A saved Random Forest learns the existing labels of that column.');
    if (!/(^|_)(ref\d*|reference|manual)(_|\d|$)/i.test(p.column)) notes.push('It learns another method\'s results, so a reference built from these suggestions cannot then be used to check that method.');
    if (p.available_examples > p.examples + p.missing_excluded) notes.push('Examples are sampled from ' + Number(p.available_examples).toLocaleString() + ' labelled cells, keeping each phenotype represented.');
    if (p.missing_excluded) notes.push(p.missing_excluded + ' examples were left out because marker measurements are missing.');
    box.querySelector('.section').replaceWith(el('div', {class: 'row section-row'}, [el('span', {class: 'section', text: 'Suggested'}),
      chip('from ' + p.column, '', 'Learned from the labels in ' + p.column), help(notes.join(' '))]));
    if (result.members) box.appendChild(el('div', {class:'hint',text:'Average model scores · '+value.scored+'/'+value.total+' cells scored'}));
    if(p.missing_markers?.length)box.appendChild(el('div',{class:'hint',text:'Using available features · missing: '+p.missing_markers.join(', '),
      title:p.note || 'Refitted using stored examples and the features measured here. Check predictions on reference cells in this tissue.'}));
    if (result.acceptedSuggestion) box.appendChild(el('div', {class: 'hint', text: result.acceptedSuggestion + ' added to the save queue.'}));
    else box.appendChild(el('div', {class: 'row wrap'}, value.choices.map(choice => el('button', {class: 'btn suggestion-choice',
      text: choice.phenotype + ' · ' + (100 * choice.probability).toFixed(1) + '%' + (result.members ? ' · Assign all' : ''),
      onclick: () => acceptSuggestion(layer, result, choice)}))));
    if (p.needs_examples?.length) box.appendChild(el('div', {class: 'hint', text:
      (p.model_id ? 'Add at least five '+(p.checked_only?'checked':'usable labelled')+' examples before retraining for: ' : 'Add a second checked example before suggestions can include: ') + p.needs_examples.join(', ') + '.'}));
  }
  card.appendChild(box);
}

async function nextUnsureCell(layer) {
  if (layer.suggestionSearch) return;
  await flushLabels();
  if (labelSaves.queue.length) return;
  const settings = suggestionSettings(layer), signature = JSON.stringify(settings);
  if (layer.unsureSettings !== signature) { layer.unsureVisited = []; layer.unsureSettings = signature; }
  const search = {settings, stopped: false}; layer.suggestionSearch = search;
  layer.unsureMessage = 'Finding cells to check...'; layer.refreshSuggestionStatus?.();
  let retry = true;
  try {
    while (!search.stopped && state.layers.includes(layer)) {
      const value = await postJSON('/cells/' + layer.token + '/suggestions', {...settings, scan: true,
        exclude: layer.unsureVisited || [], retry});
      retry = false;
      if (search.stopped) return;
      if (signature !== JSON.stringify(suggestionSettings(layer))) throw new Error('Settings changed. Start the search again.');
      layer.unsureMessage = value.message || '';
      if (value.status === 'ready') {
        if (value.label) {
          (layer.unsureVisited ||= []).push(value.label);
          layer.unsureMessage = 'Showing the least-sure unlabelled cell. ' + (value.remaining - 1).toLocaleString() + ' more to check.' +
            (value.omitted ? ' ' + value.omitted + ' cells lack complete measurements.' : '');
          setTool('off');
          await inspectReviewedCell(layer, value.label);
        } else layer.unsureMessage = 'No more unlabelled cells with complete measurements to check in this ROI.';
        return;
      }
      if (value.status !== 'preparing') return;
      layer.refreshSuggestionStatus?.();
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  } catch (error) { layer.unsureMessage = error.message; }
  finally { if (layer.suggestionSearch === search) layer.suggestionSearch = null; layer.refreshSuggestionStatus?.(); }
}

// The consensus model status is a chip; the full wording is its tooltip.
async function prepareConsensusModel(layer, chip) {
  const table = layer.table;
  const show = (text, kind, tip) => { chip.hidden = false; chip.textContent = text; chip.className = 'chip ' + kind; chip.title = tip; };
  show('Waiting', 'warn', 'The saved model will load after the image and cells.');
  await state.whenCellsReady;
  while (chip.isConnected && layer.table === table && state.layers.includes(layer)) {
    try {
      const value = await postJSON('/cells/' + layer.token + '/suggestions', {...suggestionSettings(layer), t: table, prepare: true});
      if (!chip.isConnected || layer.table !== table) return;
      if (value.status === 'ready') show('Ready', 'ok', 'Saved model ready.');
    else if (value.status === 'preparing') show('Preparing', 'warn', value.message || '');
      else show('Unavailable', 'error', value.message || '');
      if (value.status !== 'preparing') return;
    } catch (error) { show('Error', 'error', error.message); return; }
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
}

function suggestionControls(layer, body) {
  if(layer.table===layer.combinedTable&&dataView.combined?.path)restoreReferenceChoice(layer,dataView.combined.path);
  if(state.studio&&!layer.suggestionMode)layer.suggestionMode='saved_reference';
  const enabled = el('input', {type: 'checkbox', 'aria-label': 'Suggest phenotypes'}); enabled.checked = !!layer.suggestionsOn;
  const changed = () => {
    rememberReferenceChoice(layer);
    saveView();
    if (state.inspectedCell?.layer === layer) requestCellProfile(layer, state.inspectedCell.result, ++state.cardSeq);
  };
  enabled.addEventListener('change', () => { layer.suggestionsOn = enabled.checked; changed(); });
  body.appendChild(el('div', {class: 'row section-title-row'}, [
    el('label', {class: 'section-title toggle'}, [enabled, el('span', {text: 'Suggestions'})]),
    help('Shows likely phenotypes on the clicked cell. Nothing is assigned until you accept one (Enter accepts the first).')]));
  const mode = select([['consensus_rf', 'Saved consensus model'], ['saved_reference', 'Saved reference model'], ['examples', 'Selected examples']],
    layer.suggestionMode || 'consensus_rf', value => { layer.suggestionMode = value; changed(); buildPanel(); });
  mode.setAttribute('aria-label', 'Suggestion method');
  const modeTips = {
    consensus_rf: 'Learns from phenotype_ml_consensus and saved cell measurements across this table. Image marker bars are separate.',
    saved_reference: 'Uses Model training’s features, independently of displayed markers. Auto-train updates it after label edits. Missing features can use a cached model refitted to the available measurements.',
    examples: 'Learns from labelled examples in this ROI, using the measurements chosen below.'};
  const modeHelp = help(modeTips[layer.suggestionMode || 'consensus_rf']);
  const ready = el('span', {class: 'chip', role: 'status', hidden: true});
  body.appendChild(el('div', {class: 'row field'}, [el('label', {class: 'lbl', text: 'Model'}),
    el('div', {class: 'inline'}, [mode, ready]), modeHelp]));
  const showReady = (text, kind) => { ready.hidden = !text; ready.textContent = text || ''; ready.className = 'chip' + (kind ? ' ' + kind : ''); };
  if (layer.suggestionMode === 'saved_reference') {
    const choice = el('select', {'aria-label': 'Saved reference model'});
    body.appendChild(field('Saved', choice, 'The percentage is agreement with held-out reference labels.'));
    showReady('Reading...');
    getJSON('/cells/' + layer.token + '/reference_models?t=' + encodeURIComponent(layer.table || '')).then(result => {
      if (!choice.isConnected) return;
      const models = result.models, library = result.library || [];
      layer.referenceModelTable=result.table;
      const pct = value => Number.isFinite(value) ? ' · ' + Math.round(100 * value) + '%' : '';
      choice.replaceChildren(
        el('optgroup', {label: 'Saved models'}, models.map(model => el('option', {value: model.model_id,
          text: (model.auto_train && !model.imported ? 'Auto-train' : model.method) + ' · ' + model.column + pct(model.held_out_balanced_accuracy) + (!model.same_table ? ' · '+model.table : '') + (model.missing_markers?.length ? ' · ' + model.missing_markers.length + ' features missing' : '')}))),
        ...(library.length ? [el('optgroup', {label: 'Model library (picking one adds it here)'}, library.map(item => el('option', {value: 'library:' + item.entry,
          text: item.name + (item.project ? ' · from ' + item.project : '') + pct(item.accuracy) + (item.present < item.markers ? ' · ' + item.present + ' of ' + item.markers + ' markers' : '')})))] : []));
      if (!models.length && !library.length) { showReady('None trained', 'warn'); ready.title = 'Train reference models in Studio’s Phenotyping step, or share some into the model library.'; return; }
      if (!models.length) { choice.value = ''; showReady('Pick a library model', 'warn'); return; }
      const previous=layer.suggestionModelId;
      useReferenceModel(layer,models);
      if(previous!==layer.suggestionModelId)changed();
      choice.value = layer.suggestionModelId; showReady('Ready', 'ok');
    }).catch(error => { showReady('Error', 'error'); ready.title = error.message; });
    choice.addEventListener('change', async () => {
      if (choice.value.startsWith('library:')) {  // from the model library: import it into this project, then use it
        const entry = choice.value.slice(8);
        const session = new URLSearchParams(location.search).get('session');
        showReady('Adding...');
        try {
          await postJSON('/studio/' + encodeURIComponent(session) + '/model_library', {action: 'import', kind: 'phenotyping', entry});
          const picked = (await getJSON('/cells/' + layer.token + '/reference_models?t=' + encodeURIComponent(layer.table || ''))).models
            .find(model => entry.endsWith(model.model_id)) ;
          if (picked) layer.suggestionModelId = picked.model_id;
          message('Library model added to this project' + (picked ? '; another project\u2019s values are put on its training scale before it suggests.' : '.'));
        } catch (error) { message(error.message, true); }
        changed(); buildPanel(); return;
      }
      layer.suggestionModelId = choice.value; changed();
    });
  } else if (layer.suggestionMode === 'examples') {
    const columns = Array.from(new Set([...(layer.label_columns || []), ...(layer.columns || [])])).filter(c =>
      !((state.base?.info.channels) || []).includes(c) && !/^(label|batch_id|[xy]_centroid)$/i.test(c));
    if (layer.learnColumn && !columns.includes(layer.learnColumn)) columns.push(layer.learnColumn);
    const learn = select([['', 'Same as save column'], ...columns.map(c => [c, c])], layer.learnColumn || '', value => {
      layer.learnColumn = value; changed();
    });
    learn.setAttribute('aria-label', 'Learn phenotypes from');
    body.appendChild(field('Learn from', learn, 'The column whose labelled cells teach the suggestions.'));
  } else {
    prepareConsensusModel(layer, ready);
  }
  body.appendChild(field('Measure', profileChoice(layer),
    'Where marker values come from: the saved table, image pixels inside the cell outline, or the background-removed image from Tab 2.'));
  const next = el('button', {class: 'btn', text: 'Next least-sure cell',
    title: 'Opens an unlabelled cell worth checking. The first search of an ROI takes extra time.', onclick: () => {
      layer.suggestionsOn = true; enabled.checked = true; saveView(); nextUnsureCell(layer);
    }});
  const stop = el('button', {class: 'btn', text: 'Stop search', onclick: async () => {
    const search = layer.suggestionSearch; if (!search) return;
    search.stopped = true; layer.unsureMessage = 'Stopping...'; layer.refreshSuggestionStatus?.();
    try { await postJSON('/cells/' + layer.token + '/suggestions', {...search.settings, scan: true, cancel: true}); }
    catch (error) { message(error.message, true); }
  }});
  const status = el('div', {class: 'muted', role: 'status'});
  layer.refreshSuggestionStatus = () => {
    next.disabled = !!layer.suggestionSearch; stop.hidden = !layer.suggestionSearch;
    status.textContent = layer.unsureMessage || ''; status.hidden = !layer.unsureMessage;
  };
  layer.refreshSuggestionStatus();
  body.appendChild(el('div', {class: 'row'}, [next, stop])); body.appendChild(status);
}

const markerStepOrder = ['Arcsinh', 'Z1', 'Z2', 'CDF', 'NLog'];

function cellMarkerOptions() {
  if (!state.cellMarkers) {
    let saved;
    try { saved = JSON.parse(localStorage.getItem('spacemap-viewer-cell-markers')); } catch (_) { /* defaults */ }
    state.cellMarkers = {enabled:saved?.enabled === true,
      count:Number.isInteger(saved?.count) && saved.count > 0 ? saved.count : 3,
      markerSet:typeof saved?.markerSet==='string'?saved.markerSet:''};
  }
  return state.cellMarkers;
}

function cellMarkerControls() {
  const options = cellMarkerOptions();
  const box=el('div',{class:'cell-marker-controls'});
  const on = el('input', {type:'checkbox', 'aria-label':'Cell top markers'});
  on.checked = options.enabled;
  const count = el('input', {type:'number', min:'1', step:'1', required:'required',
    value:Number.isFinite(options.count) ? options.count : '', 'aria-label':'Number of cell top markers'});
  count.disabled = !options.enabled;
  const change = () => {
    options.enabled = on.checked; options.count = count.valueAsNumber;
    count.disabled = !on.checked;
    count.setCustomValidity(Number.isInteger(options.count) && options.count > 0 ? '' : 'Enter a whole number of 1 or more.');
    state.cellMarkerRequest = (state.cellMarkerRequest || 0) + 1;
    try { localStorage.setItem('spacemap-viewer-cell-markers', JSON.stringify(options)); } catch (_) { /* private mode */ }
    applyCellMarkers(state.inspectedCell);
    box.refresh();
  };
  on.addEventListener('change', change); count.addEventListener('input', change);
  box.append(el('div', {class:'row annotation-marker-controls'}, [
    el('label', {class:'inline'}, [on, el('span', {text:'Top markers'})]),
    el('label', {class:'inline'}, [el('span', {text:'Number'}), count]),
    help('Click a cell to show its highest markers. Ctrl-click several cells to rank their average transformed values. From restricts the choices to a saved image marker set. Scale is shared with the cell’s Markers panel; it changes display values, not the saved data. Colours and contrast stay as set.') ]));
  const sets=el('select',{'aria-label':'Cell top marker set'}), scales=el('div'), note=el('div',{class:'hint',role:'status'});
  sets.onchange=()=>{options.markerSet=sets.value;change();};
  box.append(field('From',sets,'All markers, or one of the named lists saved under the image’s Marker sets. The list remains fixed when the top visible markers change.'),
    field('Scale',scales,'The same transformations as the clicked cell’s Markers panel, applied in the shown order. The list filters the ranking after transformation.'),note);
  box.refresh=()=>{
    const layer=state.inspectedCell?.layer || state.layers.find(l=>l.type==='cells'&&l.visible);
    const image=state.layers.find(l=>l.type==='image'&&l.token===layer?.image_token);
    const available=image?.markerSets || {};
    sets.replaceChildren(el('option',{value:'',text:'All markers'}),...Object.keys(available).sort().map(value=>el('option',{value,text:value})),
      ...(options.markerSet&&!available[options.markerSet]?[el('option',{value:options.markerSet,text:options.markerSet+' (unavailable)'})]:[]));
    sets.value=options.markerSet;sets.disabled=!options.enabled;
    if(scales.layer!==layer){scales.replaceChildren();scales.layer=layer;if(layer)markerTransformControls(layer,scales);}
    note.textContent=options.enabled?(state.cellMarkerNote || ''):'';
  };
  box.refresh();return box;
}

document.addEventListener('spacemap-marker-sets',()=>{
  document.querySelectorAll('.cell-marker-controls').forEach(box=>box.refresh?.());
  if(cellMarkerOptions().enabled&&cellMarkerOptions().markerSet)applyCellMarkers(state.inspectedCell);
});

function cellMarkerAmount(profile, row) {
  return profile.steps?.length ? row.view : (row.signal ?? row.mean);
}

function rankedCellMarkers(profile) {
  return (profile.markers || []).filter(row => Number.isFinite(cellMarkerAmount(profile, row)))
    .slice().sort((a, b) => cellMarkerAmount(profile, b) - cellMarkerAmount(profile, a));
}

async function applyCellMarkers(inspection) {
  const options = cellMarkerOptions(), profile = inspection?.result.profile;
  if (!options.enabled || !Number.isInteger(options.count) || options.count < 1 || profile?.status !== 'ready') return;
  const image = state.layers.find(l => l.type === 'image' && l.token === inspection.layer.image_token && !l.info?.rgb);
  if (!image?.channels) return;
  const sequence = state.cardSeq, result = inspection.result;
  const request = state.cellMarkerRequest = (state.cellMarkerRequest || 0) + 1;
  const current = () => state.inspectedCell?.result === result && result.profile === profile && sequence === state.cardSeq &&
    state.layers.includes(image) && state.layers.includes(inspection.layer) && inspection.layer.visible &&
    options.enabled && request === state.cellMarkerRequest;
  if (!current()) return;
  let allowed=null;
  if(options.markerSet){
    if(!image.markerSets&&image.display_token){
      try{image.markerSets=(await getJSON('/display/'+image.display_token+'/sets')).sets;}catch(error){if(current())message(error.message,true);return;}
      if(!current())return;
    }
    allowed=new Set((image.markerSets?.[options.markerSet] || []).map(item=>item.name));
  }
  const chosen = new Set();
  for (const row of rankedCellMarkers(profile)) {
    if(allowed&&!allowed.has(row.marker))continue;
    const channel = image.channels.find(c => c.name === row.marker);
    if (channel) chosen.add(channel.index);
    if (chosen.size >= options.count) break;
  }
  state.cellMarkerNote=chosen.size ? (chosen.size<options.count?'Only '+chosen.size+' matching markers have measurements.':'') :
    options.markerSet?'No measured markers from '+options.markerSet+' are available here. The image is unchanged.':'No matching marker measurements.';
  document.querySelectorAll('.cell-marker-controls').forEach(box=>box.refresh?.());
  await showRankedMarkerChannels(image, chosen, current);
}

function markerTransformControls(layer, card) {
  const active = layer.markerTransforms || [];
  const controls = el('div', {class: 'marker-transforms', role: 'group', 'aria-label': 'Marker transformations'});
  controls.markerLayer=layer;
  const tips = {Raw:'Show the original measurement, without transformations.',
    Arcsinh:'arcsinh(value / 5): brings strong and weak signals closer together.',
    Z1:'Compare each marker with that marker in other cells of this image, using saved measurements.',
    Z2:'Compare markers with each other within this cell.',
    CDF:'Normal CDF, as in RESOLVE. Usually follows Z1 or Z2.',
    NLog:'Negative log of one minus the value, as in RESOLVE. Usually follows CDF.'};
  ['Raw', ...markerStepOrder].forEach(step => {
    const on = step === 'Raw' ? !active.length : active.includes(step);
    controls.appendChild(el('button', {type:'button', class:'btn small' + (on ? ' on' : ''),
      text:step, title:tips[step], 'aria-pressed':String(on), onclick:() => {
        const chosen = new Set(layer.markerTransforms || []);
        if (step === 'Raw') chosen.clear();
        else if (chosen.has(step)) chosen.delete(step);
        else chosen.add(step);
        layer.markerTransforms = markerStepOrder.filter(name => chosen.has(name));
        syncMarkerTransformControls(layer);
        if (chosen.has('Z1') && layer.profileSource !== 'table') {
          layer.profileSource = 'table';
          document.querySelectorAll('.profile-source').forEach(input => { input.value = 'table'; });
        }
        saveView();
        if (state.inspectedCell?.layer === layer) requestCellProfile(layer, state.inspectedCell.result, ++state.cardSeq, true);
      }}));
  });
  card.appendChild(controls);
  if (active.length > 1) card.appendChild(el('div', {class:'hint marker-transform-order', text:
    active.map((name, index) => (index+1) + '. ' + name).join(' → ')}));
}

function syncMarkerTransformControls(layer) {
  document.querySelectorAll('.marker-transforms').forEach(controls=>{
    if(controls.markerLayer!==layer)return;
    controls.querySelectorAll('button').forEach(button=>{
      const on=button.textContent==='Raw'?!layer.markerTransforms?.length:layer.markerTransforms?.includes(button.textContent);
      button.classList.toggle('on',!!on);button.setAttribute('aria-pressed',String(!!on));
    });
    const order=controls.nextElementSibling;
    if(order?.classList.contains('marker-transform-order'))order.remove();
    if(layer.markerTransforms?.length>1)controls.after(el('div',{class:'hint marker-transform-order',text:layer.markerTransforms.map((name,i)=>(i+1)+'. '+name).join(' → ')}));
  });
}

function renderCellProfile(layer, profile, card) {
  const fromTable = (profile.source || layer.profileSource || "table") === "table";
  const notes = [fromTable ? "Values saved for this cell" + (profile.table ? " in " + profile.table : "") + "." :
    (profile.image || "The image") + (profile.pixels ? " · " + Number(profile.pixels).toLocaleString() + " cell pixels" : "") + " · full resolution."];
  if (!fromTable) notes.push(profile.source === "corrected" ? "Background was already removed; signal is measured above zero." :
    "Signal is the average above each marker's min (its background level) at the time of the click; click again after changing mins. Colours and max do not affect it.");
  if (profile.view_note) notes.push(profile.view_note);
  notes.push("Raw shows the measurement; the other buttons combine in the order shown. Z1 compares all cells in this image (saved measurements). Display only: saved measurements and thresholds stay unchanged.");
  card.appendChild(el("div", { class: "row" }, [profileChoice(layer), help(notes.join(" "))]));
  markerTransformControls(layer, card);
  if (profile.message) card.appendChild(el("div", { class: "hint", role: "status", text: profile.message }));
  if (profile.status !== "ready") {
    if (profile.status === "error") card.appendChild(el("button", { class: "btn", text: "Retry", onclick: () => {
      if (state.inspectedCell) requestCellProfile(layer, state.inspectedCell.result, ++state.cardSeq);
    } }));
    return;
  }
  if (!fromTable && (profile.missing_backgrounds || []).length) card.appendChild(el("div", { class: "hint warn", text:
    "No background level for " + profile.missing_backgrounds.length + " markers: use Auto all markers, then click the cell again." }));
  const transformed = !!profile.steps?.length;
  const amount = row => cellMarkerAmount(profile, row);
  const allRows = rankedCellMarkers(profile);
  const rows = profile.showAll ? allRows : allRows.slice(0, 15);
  const largest = Math.max(...allRows.map(row => Math.abs(amount(row))), 1e-9);
  if (!rows.length) card.appendChild(el("div", { class: "hint", text: "No marker measurements are available for this cell yet." }));
  const table = el("table", { class: "cell-profile" });
  table.appendChild(el("tr", {}, [el("th", { text: "Marker" }), el("th", { text: (profile.count ? 'Mean ' : '') + (transformed ? "View value" : fromTable ? "Amount" : "Signal") })]));
  rows.forEach((row) => table.appendChild(el("tr", {
    title: (profile.count ? (row.cells || 0) + '/' + profile.count + ' cells with measurements. ' : '') + (fromTable ? "Saved measurement: " + number(row.mean) : "Average: " + number(row.mean) +
      (row.background === null ? "; background unset" : "; background: " + number(row.background)) +
      (row.positive_percent === null ? "" : "; above background: " + row.positive_percent.toFixed(1) + "% of cell pixels")),
  }, [el("td", { class: "k", text: row.marker }),
    el("td", {}, [el('div', {text: (transformed ? amount(row).toLocaleString(undefined, {maximumSignificantDigits: 4}) : number(amount(row))) + (!transformed && !fromTable && row.signal === null ? " (average)" : "")}),
      el('div', {class: 'bar' + (amount(row) < 0 ? ' neg' : ''),
        style: 'width:' + (100 * Math.abs(amount(row)) / largest) + '%'})])])));
  card.appendChild(table);
  if (allRows.length > 15) card.appendChild(el('button', {class: 'btn small', text: profile.showAll ? 'Show top 15' : 'Show all markers',
    onclick: () => { profile.showAll = !profile.showAll; showCellCard(layer, state.inspectedCell.result); }}));
}
