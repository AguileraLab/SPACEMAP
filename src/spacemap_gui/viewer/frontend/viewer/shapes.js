/* SPACEMAP viewer, part 5 of 12: drawing tools and shapes (ROIs, regions, background areas).
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- tools
function setTool(tool,selectionMode=null) {
  if (state.brushStroke) { state.brushStroke = null; state.brushCancelled = true; clearBrushTrail(); }
  // Leaving the drawing tools ends a pending "mark an example" with nothing drawn.
  if (state.drawCallback && !["rect", "polygon", "freehand"].includes(tool)) {
    const done = state.drawCallback; state.drawCallback = null; done(null);
  }
  state.selectionEdit = selectionMode;
  state.tool = tool;
  if (["off", "merge", "split"].includes(tool)) state.selectMode = false;
  if (state.mergeFirst) { state.mergeFirst = null; state.selected = null; renderShapes(); }
  drawing.points = null; drawing.start = null; drawing.hover = null;
  drawPreview();
  updateToolbar();
  buildPanel();
  updateBrushCursor();
}

function updateToolbar() {
  const tools = $("drawtools");
  const target = toolbarDrawTarget(), selectable = regionLayers().length > 0;
  tools.hidden = !target && !state.layers.some(l => l.type === 'shapes' || l.type === 'annotations');
  if (state.selectMode && !selectable) state.selectMode = false;
  if (!drawTarget() && !state.selectMode && ['rect','polygon','freehand','merge','split'].includes(state.tool)) state.tool = 'off';
  tools.querySelectorAll("button").forEach((button) => {
    const select = button.getAttribute('data-mode') === 'select', tool = button.getAttribute('data-tool');
    button.hidden = false;
    button.disabled = select ? !selectable : tool !== 'off' && !target && !(state.selectMode && selectable && ['rect','polygon','freehand'].includes(tool));
    button.classList.toggle("on", button.getAttribute("data-mode") === "select" ? !!state.selectMode : button.getAttribute("data-tool") === state.tool);
  });
  updatePolygonToolOptions();
}

function toolbarDrawTarget() {
  const current = drawTarget();
  if (current) return current;
  // A hidden assigned layer is still the user's drawing destination. Show it
  // when they explicitly choose a drawing tool; never switch training classes
  // while a requested class is loading or has failed to open.
  const assigned = state.layers.find(l => l.type === 'shapes' && l.id === state.drawInto);
  return assigned && (assigned.editable || assigned.preview?.editable) ? assigned : null;
}

function canEditPolygons(layer) {
  return layer && layer.visible && state.layers.includes(layer) && (layer.editable || layer.preview?.editable);
}

// Red detections and green examples belong to the same active artifact review.
// Prefer that pair when several artifact layers overlap in the viewer.
function artifactPickPriority(layer) {
  const target = state.layers.find(l => l.id === state.drawInto);
  const part = target?.artifact_part || target?.training_part;
  // The layer the current stage works on wins a click (marker regions over the tissue around them, say); in an
  // artifact stage, its sibling layers come next.
  return layer === target ? 2 : part && (layer.artifact_part || layer.training_part) === part ? 1 : 0;
}

function polygonToolValue(name, fallback) {
  if (state[name] === undefined) {
    try { const raw = localStorage.getItem('spacemap-' + name); state[name] = raw === null ? fallback : Number(raw); } catch (_) { state[name] = fallback; }
    if (!Number.isFinite(state[name]) || state[name] < (name === 'splitWidth' ? 0.1 : 0) || state[name] > 100000) state[name] = fallback;
  }
  return state[name];
}

function updatePolygonToolOptions() {
  let options = $('polygon-tool-options');
  if (!options) { options = el('div', {id:'polygon-tool-options'}); $('drawtools').after(options); }
  const tool = state.tool;
  options.hidden = !['split','merge'].includes(tool);
  if (options.hidden) { options.dataset.tool = ''; return; }
  if (options.dataset.tool !== tool) {
    options.dataset.tool = tool;
    const name = tool === 'split' ? 'splitWidth' : 'mergeGap', label = tool === 'split' ? 'Cut width' : 'Maximum gap';
    const input = el('input', {type:'number', min:tool === 'split' ? '0.1' : '0', max:'100000', step:'1',
      value:polygonToolValue(name, tool === 'split' ? 2 : 3), 'aria-label':label,
      title:'Full-resolution image pixels, whatever the zoom or detection scale.'});
    input.onchange = () => {
      if (!input.checkValidity() || input.value === '') { input.reportValidity(); input.value = state[name]; return; }
      state[name] = Number(input.value);
      try { localStorage.setItem('spacemap-' + name, String(state[name])); } catch (_) { /* usable without storage */ }
    };
    options.replaceChildren(el('label', {text:label + ' '}, [input]), el('span', {text:'px'}));
    if (tool === 'merge') options.appendChild(el('button', {id:'merge-selected', text:'Merge selected',
      title:'Join selected polygons in the same layer. The first polygon supplies the name.', onclick:mergeSelected}));
  }
  const button = $('merge-selected');
  if (button) button.disabled = state.brushBusy || !canEditPolygons(state.picks?.layer) || state.picks.ids.size < 2;
}

// Share Segmentation's right-click deletion with annotation previews and examples.
// Move/Select also delete a selected polygon; unselected regions keep their context menu.
function rightClickShape(point) {
  if (!point || neckAt(point) || state.tool === 'polygon' && drawing.points?.length) return null;
  const selected = [state.selected, ...(state.picks ? Array.from(state.picks.ids, id => ({layer:state.picks.layer,id})) : [])];
  const drawingTool = !state.selectMode && ['freehand','polygon','rect','merge','split'].includes(state.tool);
  for (const item of selected) {
    if (!canEditPolygons(item?.layer)) continue;
    // With the Move tool a right click opens the menu (name, zoom, delete); it deletes at once only while drawing.
    if (!drawingTool && !(item.layer.artifact_part || item.layer.training_part)) continue;
    const feature = item.layer.features.find(f => f.properties.id === item.id);
    if (feature && contains(feature, point.x, point.y)) return {...item,name:feature.properties.name};
  }
  const artifact=shapeLayers().slice().reverse().sort((a,b)=>artifactPickPriority(b)-artifactPickPriority(a)).find(l=>(l.artifact_part||l.training_part)&&l.visible&&canEditPolygons(l)&&l.features.some(f=>contains(f,point.x,point.y)));
  if(artifact){const feature=artifact.features.find(f=>contains(f,point.x,point.y));return {layer:artifact,id:feature.properties.id,name:feature.properties.name};}
  return !state.selectMode && ['freehand','polygon','rect','merge','split'].includes(state.tool) ? editableShapeAt(point) : null;
}
document.addEventListener("contextmenu", (event) => {
  const stage = $("stage");
  if (!state.viewer || !stage || !stage.contains(event.target)) return;
  const bounds = state.viewer.element.getBoundingClientRect();
  const point = toImage(new OpenSeadragon.Point(event.clientX - bounds.left, event.clientY - bounds.top));
  const hit = rightClickShape(point);
  if (!hit) {
    if (!neckAt(point) && !state.selectMode && ['freehand','polygon','rect','merge','split'].includes(state.tool)) event.preventDefault();
    return;
  }
  event.preventDefault();event.stopImmediatePropagation();closeContextMenu();
  if(hit.layer.artifact_part||hit.layer.training_part){
    const ids=state.picks?.layer===hit.layer&&state.picks.ids.has(hit.id)?Array.from(state.picks.ids):[hit.id];
    const part=hit.layer.artifact_part||hit.layer.training_part,role=hit.layer.training_part?'positive':'negative';
    const label=role==='positive'?'artifact':part==='blur'?'sharp tissue':part==='lint'?'good glass':'good tissue';
    const act=(text,run)=>el('button',{type:'button',text,onclick:()=>{closeContextMenu();run();}});
    const menu=el('div',{id:'ctxmenu',role:'menu'},[
      act(ids.length>1?'Delete selected ('+ids.length+')':'Delete',()=>deleteRegions(hit.layer,new Set(ids))),
      act((ids.length>1?'Mark selected as ':'Mark as ')+label,()=>convertArtifactSelection(hit.layer,ids,role))]);
    menu.style.left=Math.min(event.clientX,window.innerWidth-250)+'px';
    menu.style.top=Math.min(event.clientY,window.innerHeight-90)+'px';document.body.append(menu);
  }else deleteShape(hit.layer, hit.id);
}, true);

document.addEventListener("keydown", (event) => {
  const target = event.target || {};
  const polygonBox = target.tagName === 'INPUT' && target.type === 'checkbox' && target.closest('.region-selection, .region-list');
  if (target.isContentEditable || /SELECT|TEXTAREA/.test(target.tagName || "") ||
      (target.tagName === 'INPUT' && !polygonBox)) return;
  if (event.key === "Escape") {
    if (state.pointCallback) { const done = state.pointCallback; state.pointCallback = null; done(null); }
    drawing.points = null; drawing.start = null; drawing.hover = null; drawPreview();
    if (state.mergeFirst) { state.mergeFirst = null; state.selected = null; renderShapes(); }
    if (state.drawCallback) setTool("off");
  } else if (event.key === "Enter" && state.tool === "polygon") {
    finishPolygon();
  } else if (event.key === 'Enter' && target.tagName !== 'BUTTON' && !event.repeat) {
    const inspected = state.inspectedCell;
    const choice = inspected?.result.suggestions?.choices?.[0];
    if (choice && inspected.layer.suggestionsOn) {
      event.preventDefault(); acceptSuggestion(inspected.layer, inspected.result, choice);
    }
  } else if ((event.ctrlKey || event.metaKey) && ["z", "Z", "y", "Y"].includes(event.key)) {
    // Undo and redo the last drawn or deleted shape of the layer being drawn on (or the regions last changed).
    const layer = state.lastEdited && state.layers.includes(state.lastEdited) ? state.lastEdited : drawTarget();
    if (!layer || layer.callback || layer.selection) return;
    event.preventDefault();
    if (event.key.toLowerCase() === "y" || event.shiftKey) redoShapes(layer); else undoShapes(layer);
  } else if (event.key === "Backspace" || event.key === "Delete") {
    if (state.tool === "polygon" && drawing.points && drawing.points.length) { drawing.points.pop(); drawPreview(); }
    else if (state.picks && state.picks.ids.size) deleteRegions(state.picks.layer, new Set(state.picks.ids));
    else if (state.selected) deleteShape(state.selected.layer, state.selected.id);
    event.preventDefault();
  }
});

// ---------------------------------------------------------------- shapes (drawn regions)
const SVG_NS = "http://www.w3.org/2000/svg";
function shapeLayers() { return state.layers.filter((l) => l.type === "shapes"); }
// New shapes go to the topmost editable layer: the one most recently added (Studio's ROI layer).
function drawTarget() {
  if (state.drawCallback) return { callback: true };
  if (state.selectionEdit) return { selection: true };
  // Studio says which layer its stage draws into (drawn artifacts, marker regions); else the topmost editable layer,
  // regions last so a drawing never lands among them by chance.
  const open = shapeLayers().reverse().filter((l) => (l.editable || l.preview?.editable) && l.visible);
  if (state.drawInto) return open.find(l => l.id === state.drawInto) || null;
  return open.find((l) => l.id === state.drawInto) || open.find((l) => !l.regions) || open[0] || null;
}

async function loadShapes(layer) {
  layer.loadingShapes = true;
  layer.status = "Loading polygons…";
  shapeStatus(layer);
  try {
    const result = await shapeRequest(layer);
    clearShapePreview(layer, false);
    layer.features = result.features || [];
    layer.revision = result.revision;
    for (const key of ['regions','annotation_kind','saved_file']) if (result[key] !== undefined) layer[key] = result[key];
    layer.editable = layer.editable !== false && result.editable !== false;
    layer.history = []; layer.future = [];
    layer.status = loadedShapesText(layer);
    layer.shapesLoadError = null;
    layer.saveError = null;
    return true;
  } catch (error) {
    layer.shapesLoadError = error.message;
    layer.status = "Could not load polygons: " + error.message;
    message(layer.status, true);
    return false;
  } finally { layer.loadingShapes = false; shapeStatus(layer); }
}

async function showTissuePreview(found) {
  if (!found.layer || !Array.isArray(found.features)) return false;
  if (!state.layers.some(l => l.type === 'image' && l.token === found.layer.image_token)) return false;
  let layer = state.layers.find(l => l.id === found.layer.id);
  if (!layer) layer = await addLayerToView({...found.layer});
  if (!layer || !state.layers.includes(layer) || layer.brushBusy) return false;
  await flushShapeSave(layer);
  if (!state.layers.includes(layer) || layer.image_token !== found.layer.image_token) return false;
  if (!layer.preview) layer.preview = {features:layer.features, editable:layer.editable, status:layer.status,
    history:layer.history, future:layer.future};
  layer.history = []; layer.future = [];
  layer.features = found.features; layer.editable = false; layer.regions = true;
  layer.visible = true; layer.open = true;
  if (layer.annotation_kind === 'tissue') state.tissueLayerId = layer.id;
  (state.layerGroups || (state.layerGroups = {})).annotations = true;
  layer.status = tissuePreviewText(layer);
  state.selected = null; state.picks = null;
  showPreview('seg', null); state.segPreview = null;
  renderShapes(); buildPanel(); updateToolbar();
  return true;
}

function tissuePreviewText(layer) {
  return 'Preview — ' + layer.features.length + ' polygons, not saved. Select unwanted pieces and delete them, then Save polygons.';
}

async function saveTissuePreview(layer) {
  if (!layer.preview?.editable || layer.preview.saving || layer.brushBusy) return false;
  if (layer.preview.features?.length && !window.confirm('Replace the saved ' + layer.name + ' polygons? The previous version stays in History.')) return false;
  const draft = layer.preview;
  draft.saving = true; layer.saveError = null; layer.status = 'Saving polygons…'; buildPanel(); refreshRegionList();
  try {
    layer.saveBusy = shapeRequest(layer, '/save', {features:layer.features, revision:layer.revision});
    const result = await layer.saveBusy;
    if (layer.preview !== draft) return;
    const before = JSON.stringify(draft.features || []);
    draft.saving = false;
    clearShapePreview(layer, false);
    layer.features = result.features; layer.revision = result.revision;
    layer.history = [...(layer.history || []), before]; layer.future = [];
    layer.status = 'Polygons saved.'; state.lastEdited = layer;
    if (state.studio?.regionsChanged) state.studio.regionsChanged(layer);
    return true;
  } catch (error) {
    layer.saveError = error.message;
    layer.status = 'Could not save polygons: ' + error.message; message(layer.status, true);
    return false;
  } finally {
    draft.saving = false; layer.saveBusy = null;
    renderShapes(); buildPanel(); refreshRegionList(); updateToolbar();
  }
}

function clearShapePreview(layer, redraw = true) {
  if (!layer.preview || layer.preview.saving || layer.brushBusy) return;
  const {saving, ...original} = layer.preview;
  Object.assign(layer, original); layer.preview = null;
  layer.saveError = null;
  if (state.selected?.layer === layer) state.selected = null;
  if (state.picks?.layer === layer) state.picks = null;
  if (redraw) { renderShapes(); buildPanel(); updateToolbar(); refreshRegionList(); }
}

function loadedShapesText(layer) {
  const count = (layer.features || []).length;
  return count ? count + (count === 1 ? ' polygon loaded.' : ' polygons loaded.') :
    layer.annotation_kind === 'tissue' ? 'No saved polygons. Run Detect tissue to create them.' : 'No saved shapes yet.';
}

// Polygon transfers are short requests, unlike detection jobs. A lost response
// must not leave the layer saying Saving forever. Aborting the request does not
// undo a possible server save: Reload saved reads its actual, revision-checked state.
async function shapeRequest(layer, suffix = '', payload) {
  const controller = new AbortController();
  // A save can be slow while a job keeps the machine busy: wait longer for it than for a read.
  const timer = setTimeout(() => controller.abort(), suffix === '/save' ? 120000 : 30000);
  try {
    const path = '/shapes/' + layer.token + suffix;
    return await (payload === undefined ? getJSON(path, {signal:controller.signal}) : postJSON(path, payload, {signal:controller.signal}));
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The server did not reply. Use Reload saved to check what was saved.');
    throw error;
  } finally { clearTimeout(timer); }
}

async function reloadSavedShapes(layer) {
  if (layer.saveBusy || layer.brushBusy || layer.loadingShapes) return;
  if (layer.savePending && !window.confirm('Reload the saved polygons? Changes still waiting in this window will be replaced by the saved version.')) return;
  clearTimeout(layer.saveTimer);
  if (await loadShapes(layer)) {
    layer.savePending = false;
    if (state.selected?.layer === layer) state.selected = null;
    if (state.picks?.layer === layer) state.picks = null;
    state.cardSeq += 1; $('cellcard').hidden = true;
  }
  renderShapes(); buildPanel(); refreshRegionList();
}

function shapeStorageControls(layer, body) {
  body.appendChild(el('div', {class:'hint', 'data-shape-status':layer.id, text:layer.status || ''}));
  if (layer.preview?.editable && body.dataset.saveInFooter !== 'true') body.appendChild(el('button', {class:'btn primary small', text:'Save polygons',
    disabled:layer.preview.saving ? '' : null, onclick:() => saveTissuePreview(layer)}));
  if (layer.local) return;
  const reload = el('button', {class:'btn small', text:'Reload saved', 'data-shape-reload':layer.id,
    title:'Read the saved polygons without moving the image.', onclick:() => reloadSavedShapes(layer)});
  const retry = el('button', {class:'btn small', text:'Retry save', 'data-shape-retry':layer.id, hidden:!layer.savePending,
    onclick:() => flushShapeSave(layer).catch(() => {})});
  reload.disabled = !!(layer.saveBusy || layer.brushBusy || layer.loadingShapes);
  retry.disabled = !!(layer.saveBusy || layer.brushBusy || layer.loadingShapes);
  body.appendChild(el('div', {class:'row'}, [reload, retry]));
  if (layer.saved_file) {
    const file = el('details', {class:'shape-saved-file'}, [el('summary', {text:'Saved file'}),
      el('div', {class:'hint', text:layer.saved_file, style:'overflow-wrap:anywhere;user-select:text'}),
      el('div', {class:'hint', text:'The × removes this layer from the view. Clearing polygons changes the saved file; History keeps earlier versions.'})]);
    if (layer.annotation_kind) file.appendChild(el('button', {class:'btn small', text:'Clear saved polygons',
      disabled:!layer.editable || !layer.features?.length || layer.brushBusy || layer.saveBusy ? '' : null,
      onclick:async () => {
        if (!layer.editable || layer.preview || layer.brushBusy || layer.saveBusy) return;
        if (!window.confirm('Clear all saved polygons in ' + layer.name + '? Undo and History can bring them back.')) return;
        deleteRegions(layer, new Set(layer.features.map(f => f.properties.id)));
        try { await flushShapeSave(layer); } catch (_) {}
      }}));
    body.appendChild(file);
  }
}

function ensureOverlay() {
  if (state.svg) return;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("id", "shapes");
  const root = document.createElementNS(SVG_NS, "g");
  svg.appendChild(root);
  $("stage").insertBefore(svg, $("toolbar"));
  state.svg = svg; state.svgRoot = root;
}

function placeOverlay() {
  const item = baseItem();
  if (!state.svgRoot || !item) return;
  const origin = item.imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0));
  const unit = item.imageToViewerElementCoordinates(new OpenSeadragon.Point(1000, 0));
  state.overlayScale = (unit.x - origin.x) / 1000;
  state.svgRoot.setAttribute("transform", "translate(" + origin.x + "," + origin.y + ") scale(" + state.overlayScale + ")");
  drawLabelPreview();
  drawCellOutline();
  drawSegPreview();
}

// The clicked cell's boundary, so it is clear which cell the card describes.
function drawCellOutline() {
  if (!state.svgRoot) return;
  if (!state.cellOutline || !state.cellOutline.length) {
    if (state.cellOutlineEl) { state.cellOutlineEl.remove(); state.cellOutlineEl = null; }
    return;
  }
  if (!state.cellOutlineEl) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("class", "cell-outline");
    path.setAttribute("vector-effect", "non-scaling-stroke");
    state.cellOutlineEl = path;
  }
  state.cellOutlineEl.setAttribute("d", state.cellOutline.map((ring) =>
    ring.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join("") + "Z").join(""));
  if (!state.cellOutlineEl.isConnected) state.svgRoot.appendChild(state.cellOutlineEl);
}

// Studio's previews drawn over the image (cells of a segmentation test, tissue,
// an artifact's live mask, drawn examples), each under its own name.
function drawSegPreview() {
  if (!state.svgRoot) return;
  const previews = state.previews || (state.previews = {});
  const elements = state.previewEls || (state.previewEls = {});
  Object.keys(elements).forEach((key) => {
    if (!previews[key]) { elements[key].forEach((node) => node.remove()); delete elements[key]; }
  });
  Object.entries(previews).forEach(([key, preview]) => {
    const nodes = elements[key] || (elements[key] = [document.createElementNS(SVG_NS, "path"), document.createElementNS(SVG_NS, "path")]);
    if (key.startsWith('pieces-') && !state.layers.some(l => l.id === (state.tissueLayerId || 'tissue-regions') && l.visible && l.features === state.piecesPreviewFeatures)) { nodes.forEach(n => n.remove()); return; }
    const [outline, box] = nodes;
    outline.setAttribute("class", "seg-preview" + (preview.kind ? " " + preview.kind : ""));
    outline.setAttribute("vector-effect", "non-scaling-stroke");
    if (preview.colour) { outline.style.stroke = preview.colour; outline.style.fill = preview.colour + "33"; }
    outline.setAttribute("d", (preview.outlines || []).map((ring) =>
      ring.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join("") + "Z").join(""));
    if (!outline.isConnected) state.svgRoot.appendChild(outline);
    if (preview.bounds) {
      const [x, y, w, h] = preview.bounds;
      box.setAttribute("class", "seg-preview-box");
      box.setAttribute("vector-effect", "non-scaling-stroke");
      box.setAttribute("d", "M" + x + " " + y + "h" + w + "v" + h + "h" + (-w) + "Z");
      if (!box.isConnected) state.svgRoot.appendChild(box);
    } else if (box.isConnected) box.remove();
  });
}

function showPreview(name, preview) {
  if (name === null) { state.neckReview = null; state.previews = {}; drawSegPreview(); return; }  // Studio clears every overlay at once.
  (state.previews || (state.previews = {}))[name] = preview && preview.outlines ? preview : null;
  if (name.startsWith('pieces-') && preview?.outlines) state.piecesPreviewFeatures = state.layers.find(l => l.id === (state.tissueLayerId || 'tissue-regions'))?.features;
  if (!state.previews[name]) delete state.previews[name];
  drawSegPreview();
}

function showNeckReview(review) {
  state.neckReview = review;
  showPreview('pieces-necks', review?.regions.length ? {outlines:review.regions.flatMap(n => polygonsOf(n.geometry).flat()), colour:'#ff4848', kind:'neck'} : null);
}
function neckAt(point) {
  return point && state.layers.some(l => l.id === (state.tissueLayerId || 'tissue-regions') && l.visible && l.features === state.piecesPreviewFeatures) && state.neckReview?.regions.find(n => contains({geometry:n.geometry}, point.x, point.y));
}
function neckMenuItems(point, act) {
  const neck = neckAt(point), review = state.neckReview;
  return neck ? [el('div', {class:'ctx-title', text:'Detected neck · ' + neck.width_um.toFixed(1) + ' µm'}),
    act('Approve cut', () => review.approve(neck), state.brushBusy || review.busy),
    act('Reject suggestion', () => review.reject(neck), state.brushBusy || review.busy)] : [];
}

// Studio's raster overlays (a pixel class map), one image laid over the whole base image.
function showRaster(name, raster) {
  const rasters = state.rasters || (state.rasters = {});
  const old = rasters[name];
  if (raster && raster.opacity !== undefined && !raster.url && old) {
    old.opacity = raster.opacity; if (old.item) old.item.setOpacity(raster.opacity); return;
  }
  if (!raster) { if (old && old.item && state.viewer) state.viewer.world.removeItem(old.item); delete rasters[name]; return; }
  const item = baseItem();
  if (!item || !state.viewer) return;
  const bounds = item.getBounds();
  const entry = { opacity: raster.opacity === undefined ? (old ? old.opacity : 0.6) : raster.opacity };
  rasters[name] = entry;
  state.viewer.addTiledImage({
    tileSource: { type: "image", url: raster.url, buildPyramid: false },
    x: bounds.x, y: bounds.y, width: bounds.width, opacity: entry.opacity,
    success: (event) => {
      if (rasters[name] !== entry) { state.viewer.world.removeItem(event.item); return; }
      entry.item = event.item;
      if (old && old.item) state.viewer.world.removeItem(old.item);
    },
  });
}

// Studio's "mark an example": the next drawn shape goes to Studio instead of a file.
function drawExample(tool) {
  return new Promise((resolve) => {
    if (state.drawCallback) state.drawCallback(null);
    state.drawCallback = resolve;
    setTool(tool || "polygon"); updateToolbar();
    message("Click the corners of the area; double-click or press Enter to finish. Esc cancels.");
  });
}

async function outlineCell(layer, label, point) {
  state.cellOutline = null;
  drawCellOutline();
  if (!label) return;
  const seq = state.cardSeq;
  const where = point && Number.isFinite(point.x) && Number.isFinite(point.y) ? "&x=" + point.x + "&y=" + point.y : "";
  try {
    const result = await getJSON("/cells/" + layer.token + "/outline?label=" + label + where + "&t=" + encodeURIComponent(layer.table || ""));
    if (seq !== state.cardSeq) return;
    state.cellOutline = result.rings || null;
    if (state.inspectedCell?.result?.label === label) state.inspectedCell.result.rings = result.rings || [];
    scheduleSplitMarkers();
    ensureOverlay();
    placeOverlay();
  } catch (error) { /* the card still describes the cell; the outline only helps find it */ }
}

function polygonsOf(geometry) {
  if (!geometry) return [];
  if (geometry.type === "Polygon") return [geometry.coordinates];
  if (geometry.type === "MultiPolygon") return geometry.coordinates;
  if (geometry.type === "GeometryCollection") return [].concat(...(geometry.geometries || []).map(polygonsOf));
  return [];
}
function linesOf(geometry) {
  if (!geometry) return [];
  if (geometry.type === "LineString") return [geometry.coordinates];
  if (geometry.type === "MultiLineString") return geometry.coordinates;
  if (geometry.type === "GeometryCollection") return [].concat(...(geometry.geometries || []).map(linesOf));
  return [];
}
function pathOf(geometry) {
  const line = (points) => points.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join("");
  return polygonsOf(geometry).map((rings) => rings.map((ring) => line(ring) + "Z").join("")).join("") +
    linesOf(geometry).map(line).join("");
}
function shapeColor(feature) {
  const chosen = (feature.properties || {}).colour;  // a name's colour chosen in Layers (tissue keeps its class)
  if (typeof chosen === "string" && /^#[0-9a-f]{6}$/i.test(chosen)) return chosen;
  const color = ((feature.properties || {}).classification || {}).color;
  return Array.isArray(color) ? "#" + color.slice(0, 3).map((v) => Number(v).toString(16).padStart(2, "0")).join("") : "#ffd600";
}
function rgbOf(color) { const h = hex(color); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }

function renderShapes() {
  if (!state.svgRoot) return;
  state.svgRoot.innerHTML = "";
  shapeLayers().forEach((layer) => {
    if (!layer.visible) return;
    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("data-layer", layer.id);
    group.setAttribute("opacity", String(layer.opacity));
    (layer.features || []).forEach((feature) => {
      if (layer.hiddenNames && !featureShown(layer, feature)) return;  // a name row hidden in Layers
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", pathOf(feature.geometry));
      const color = shapeColor(feature);
      path.setAttribute("stroke", color);
      path.setAttribute("fill", color);
      if (layer.training_part || layer.artifact_part) path.style.stroke = color;
      const selected = (state.selected && state.selected.id === feature.properties.id) || isPicked(layer, feature.properties.id);
      path.setAttribute("class", "shape" + (selected ? " selected" : ""));
      group.appendChild(path);
    });
    state.svgRoot.appendChild(group);
  });
  state.preview = document.createElementNS(SVG_NS, "path");
  state.preview.setAttribute("class", "shape preview");
  state.svgRoot.appendChild(state.preview);
  drawPreview();
  placeOverlay();
}

function drawPreview() {
  if (!state.preview) return;
  const target = drawTarget(), cls = (target?.training_part||target?.artifact_part) && currentClass(target);
  state.preview.style.stroke = cls ? cls.color : '';
  state.preview.style.fill = cls ? cls.color : '';
  let d = "";
  if (drawing.points && drawing.points.length) {
    const points = drawing.points.concat(state.tool === "polygon" && drawing.hover ? [drawing.hover] : []);
    d = points.map((p, i) => (i ? "L" : "M") + p.x + " " + p.y).join("") + (["polygon", "split"].includes(state.tool) ? "" : "Z");
  }
  state.preview.setAttribute("d", d);
}

function screenOf(point) {
  const item = baseItem();
  return item ? item.imageToViewerElementCoordinates(new OpenSeadragon.Point(point.x, point.y)) : null;
}

function polygonClick(position) {
  const point = toImage(position);
  if (!point || !(drawTarget() || state.selectMode && regionLayers().length)) return;
  drawing.points = drawing.points || [];
  const points = drawing.points;
  if (points.length) {
    const last = screenOf(points[points.length - 1]);
    if (last && Math.hypot(last.x - position.x, last.y - position.y) < 4) return;  // the second click of a double-click
  }
  if (points.length >= 3) {
    const first = screenOf(points[0]);
    if (first && Math.hypot(first.x - position.x, first.y - position.y) < 10) { finishPolygon(); return; }
  }
  points.push(point);
  drawPreview();
}

function finishPolygon() {
  const points = drawing.points;
  drawing.points = null; drawing.hover = null;
  drawPreview();
  if (points && points.length >= 3) addShape(points);
}

const SHAPE_COLORS = ["#ffd600", "#00e5ff", "#ff4fd8", "#7cff6b", "#ff8c00", "#4c8dff", "#ffffff", "#ff3b3b"];

function currentClass(layer) {
  const classes = layer.classes || [];
  return classes.find((c) => c.name === layer.default_class) || classes[0] || null;
}

function newId() { return Math.random().toString(16).slice(2, 10); }

function addShape(points) {
  if (state.drawCallback) {
    const done = state.drawCallback;
    state.drawCallback = null; setTool("off"); updateToolbar();
    done(points.map((point) => [point.x, point.y]));
    return;
  }
  if (state.selectMode) { selectInside(points, !!state.shiftDown); return; }
  if (state.selectionEdit) {
    runCommand('refine_selection',{points:points.map(point=>[point.x,point.y]),mode:state.selectionEdit})
      .then(result=>state.studio?.showResult(result)).catch(error=>message(error.message,true));
    return;
  }
  const layer = drawTarget();
  if (!layer) return;
  const ring = points.map((p) => [Math.round(p.x * 10) / 10, Math.round(p.y * 10) / 10]);
  ring.push(ring[0].slice());
  if (ringArea(ring) < 4) { message("That shape is too small; drag further or zoom in.", true); return; }
  const cls = currentClass(layer);
  const count = (layer.features || []).length + 1;
  const name = (cls ? cls.name : "Shape") + " " + count;
  const color = cls ? cls.color : SHAPE_COLORS[(count - 1) % SHAPE_COLORS.length];
  const properties = { id: newId(), name: name, label: name, classification: { name: cls ? cls.name : name, color: rgbOf(color) } };
  if(layer.artifact_part)properties.source='manual_artifact';
  if (layer.training_part) properties.example_role = cls?.name === 'Artifact' ? 'positive' : 'negative';
  pushHistory(layer); state.lastEdited = layer;
  layer.features.push({ type: "Feature", geometry: { type: "Polygon", coordinates: [ring] }, properties: properties });
  state.selected = { layer: layer, id: properties.id };
  if (layer.regions) state.picks = {layer, ids:new Set([properties.id])};
  saveShapes(layer);
  renderShapes();
  buildPanel();
  message(name + ": " + areaText(ringArea(ring)) + ".");
}

function deleteShape(layer, id) {
  if (!layer || !(layer.editable || layer.preview?.editable)) return;
  if (!(layer.features || []).some(f => f.properties.id === id)) return;
  deleteRegions(layer, new Set([id]));
}

// Merge and Split replace shapes with the server's result; Ctrl+Z undoes them like any other change.
function replaceShapes(layer, removed, polygons, properties) {
  state.lastEdited = layer;
  const gone = new Set(removed.map((f) => f.properties.id));
  const { simplified, _studio_geometry, ...kept } = properties;  // the new outline is exact, not a simplified copy
  const made = polygons.map((coordinates, index) => ({ type: "Feature", geometry: { type: "Polygon", coordinates },
    properties: index ? { ...kept, id: newId(), name: kept.name + " part " + (index + 1), label: kept.name + " part " + (index + 1) } : kept }));
  pushHistory(layer);
  layer.features = layer.features.flatMap((f) => f === removed[0] ? made : gone.has(f.properties.id) ? [] : [f]);
  state.selected = null;
  saveShapes(layer);
  renderShapes();
  buildPanel();
  refreshRegionList();
  return made;
}

// Merge uses checked polygons, or two successive clicks on polygons in one layer.
async function mergeClick(point) {
  const hit = editableShapeAt(point), first = state.mergeFirst;
  if (!hit) { message("Click inside a shape to merge it."); return; }
  const find = (layer, id) => layer.features.find((f) => f.properties.id === id);
  if (!first || first.layer !== hit.layer || !find(first.layer, first.id)) {
    state.mergeFirst = hit; state.selected = { layer: hit.layer, id: hit.id }; renderShapes();
    message((hit.name || "Shape") + " picked; now click the shape to join it with (Esc cancels).");
    return;
  }
  state.mergeFirst = null; state.selected = null;
  if (first.id === hit.id) { renderShapes(); message("Merge cancelled."); return; }
  await applyPolygonEdit(hit.layer, [first.id, hit.id], {op:'merge', gap:polygonToolValue('mergeGap',3)});
}

async function mergeSelected() {
  const picks = state.picks;
  if (!canEditPolygons(picks?.layer) || picks.ids.size < 2) return;
  await applyPolygonEdit(picks.layer, Array.from(picks.ids), {op:'merge', gap:polygonToolValue('mergeGap',3)});
}

// The selected polygon wins; otherwise find the one under the line's middle.
async function splitAlong(start, end) {
  if (!start || !end || Math.hypot(end.x - start.x, end.y - start.y) < 2) { message("Drag a line across a shape to cut it."); return; }
  const hit = brushTarget() || editableShapeAt({ x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 });
  if (!hit) { message("Drag the line across the middle of a shape to cut it."); return; }
  await applyPolygonEdit(hit.layer, [hit.id], {op:'split', width:polygonToolValue('splitWidth',2), line:[[start.x,start.y],[end.x,end.y]]});
}

async function applyPolygonEdit(layer, ids, operation) {
  if (!canEditPolygons(layer) || state.brushBusy || layer.preview?.saving) return;
  state.brushBusy = true; layer.brushBusy = true;
  shapeStatus(layer);
  message(operation.op === 'neck' ? 'Applying neck cut…' : operation.op === 'split' ? 'Cutting polygon…' : 'Merging polygons…');
  try {
    await flushShapeSave(layer);
    const targets = layer.features.filter(f => ids.includes(f.properties.id));
    if (targets.length !== ids.length) throw Error('These polygons changed. Select them again.');
    let selected;
    if (layer.local || layer.preview) {
      const result = await postJSON('/shapes/edit', {...operation, image_token:layer.image_token, source_token:layer.token, preview:true, revision:layer.revision, geometries:targets.map(f => f.geometry),
        ...(operation.op === 'neck' ? {features:targets} : {})});
      if (!state.layers.includes(layer) || targets.some(f => !layer.features.includes(f))) return;
      if (operation.op === 'neck') {
        pushHistory(layer); layer.features = result.features; selected = result.selected; saveShapes(layer);
      } else selected = replaceShapes(layer, targets, result.geometries.map(g => g.coordinates), targets[0].properties).map(f => f.properties.id);
    } else {
      const result = await shapeRequest(layer, '/edit', {...operation, ids, revision:layer.revision});
      pushHistory(layer);
      layer.features = result.features; layer.revision = result.revision; layer.status = 'Saved.';
      selected = result.selected;
    }
    if (!state.layers.includes(layer)) return;
    state.lastEdited = layer; state.mergeFirst = null;
    state.picks = {layer, ids:new Set(selected)};
    state.selected = selected.length ? {layer, id:selected[0]} : null;
    state.cardSeq += 1; $('cellcard').hidden = true;
    renderShapes(); buildPanel(); refreshRegionList(); shapeStatus(layer);
    message((operation.op === 'neck' ? 'Neck cut approved.' : operation.op === 'split' ? 'Polygon split.' : 'Polygons merged.') + ' Ctrl+Z undoes this change.');
    return true;
  } catch (error) { message(error.message, true); }
  finally { state.brushBusy = false; layer.brushBusy = false; buildPanel(); refreshRegionList(); shapeStatus(layer); }
}

async function convertArtifactSelection(layer,ids,role) {
  if(layer.brushBusy||state.brushBusy||!state.layers.includes(layer)||!(layer.editable||layer.preview?.editable)||!ids.length)return;
  const part=layer.artifact_part||layer.training_part;
  const image=state.layers.find(l=>l.type==='image'&&l.token===layer.image_token);
  if(!part||!image)return;
  let red,good;
  state.brushBusy=true;layer.brushBusy=true;
  try{
    await flushAllShapes();
    const session=new URLSearchParams(location.search).get('session');
    const route='/studio/'+encodeURIComponent(session)+'/artifact_review/'+part;
    const current=()=>state.layers.includes(image)&&state.layers.includes(layer);
    red=state.layers.find(l=>l.id==='artifact-'+part&&l.image_token===image.token);
    if(!red){
      const catalogue=await getJSON(route+'/state');if(!current())return;
      red=await addLayerToView({...catalogue.layer,visible:true});
    }
    const opened=await postJSON(route+'/open_examples',{image_token:image.token,artifact_preview:!!red.preview});
    if(!current())return;
    good=state.layers.find(l=>l.id===opened.layer.id&&l.image_token===image.token);
    if(!good)good=await addLayerToView({...opened.layer,visible:true});
    if(opened.combined_red){if(!red.preview)await loadShapes(red);await loadShapes(good);}
    if(!current()||!state.layers.includes(red)||!state.layers.includes(good))return;
    if([red,good].some(l=>l.loadingShapes||l.saveError||l.preview?.saving))throw Error('Wait for artifact polygons to finish loading or saving.');
    red.brushBusy=true;good.brushBusy=true;buildPanel();refreshRegionList();
    const other=layer===red?good:red;
    const before={features:JSON.stringify(layer.features),other:{id:other.id,features:JSON.stringify(other.features)}};
    const result=await postJSON(route+(role==='negative'?'/mark_good':'/mark_artifact'),{image_token:image.token,ids,revision:red.revision,
      example_revision:good.revision,...(red.preview?{preview_features:red.features}:{})});
    if(!current()||!state.layers.includes(red)||!state.layers.includes(good))return;
    layer.history=layer.history||[];layer.history.push(before);if(layer.history.length>100)layer.history.shift();layer.future=[];other.future=[];
    red.features=result.artifact.features;red.revision=result.artifact.revision;
    good.features=result.examples.features;good.revision=result.examples.revision;
    (role==='negative'?good:red).visible=true;
    red.status=red.preview?tissuePreviewText(red):'Saved.';good.status='Saved.';
    state.lastEdited=layer;state.selected=null;state.picks=null;
    const label=role==='positive'?'artifact':part==='blur'?'sharp tissue':part==='lint'?'good glass':'good tissue';
    message(result.converted+' polygon'+(result.converted===1?'':'s')+' marked as '+label+'. Ctrl+Z undoes this change.');
  }catch(error){message(error.message,true);}
  finally{for(const item of [layer,red,good])if(item)item.brushBusy=false;state.brushBusy=false;renderShapes();buildPanel();refreshRegionList();shapeStatus(layer);}
}

function pairedHistory(layer,entry,destination) {
  if(typeof entry==='string')return false;
  const other=state.layers.find(l=>l.id===entry.other.id);
  if(!other)throw Error('Open both artifact and good-tissue layers before undoing this change.');
  destination.push({features:JSON.stringify(layer.features),other:{id:other.id,features:JSON.stringify(other.features)}});
  layer.features=JSON.parse(entry.features);other.features=JSON.parse(entry.other.features);
  saveShapes(other);return true;
}

function pushHistory(layer) {
  layer.history = layer.history || [];
  layer.history.push(JSON.stringify(layer.features || []));
  if (layer.history.length > 100) layer.history.shift();
  layer.future = [];  // a new change ends what could be redone
}

function undoShapes(layer) {
  if (layer.preview?.saving) return;
  if (layer.brushBusy) { message("Finishing the brush stroke…"); return; }
  if (!layer.history || !layer.history.length) { message("Nothing to undo."); return; }
  layer.future = layer.future || [];
  const entry=layer.history.pop();
  if(!pairedHistory(layer,entry,layer.future)){layer.future.push(JSON.stringify(layer.features || []));layer.features=JSON.parse(entry);}
  if (state.selected && state.selected.layer === layer && !layer.features.some((f) => f.properties.id === state.selected.id)) state.selected = null;
  saveShapes(layer);
  renderShapes();
  buildPanel();
  refreshRegionList();
  message("Undone. Ctrl+Y redoes it.");
}

function redoShapes(layer) {
  if (layer.preview?.saving) return;
  if (layer.brushBusy) return;
  if (!layer.future || !layer.future.length) { message("Nothing to redo."); return; }
  layer.history = layer.history || [];
  const entry=layer.future.pop();
  if(!pairedHistory(layer,entry,layer.history)){layer.history.push(JSON.stringify(layer.features || []));layer.features=JSON.parse(entry);}
  saveShapes(layer);
  renderShapes();
  buildPanel();
  refreshRegionList();
}

// TLS drafts use the same selection, Shift/Alt brush, Merge, Split and Undo as annotations.
// Their geometry stays local until Spatial analysis saves and retags the cohort.
function tlsBoundaryEditor(action, payload = {}) {
  let layer = state.layers.find(l => l.tlsDraft);
  const copy = value => JSON.parse(JSON.stringify(value));
  const features = items => items.map((f,i) => ({...copy(f), properties:{...copy(f.properties || {}),
    id:newId(), name:'TLS ' + (i+1), label:'TLS ' + (i+1), classification:{name:'TLS candidate',color:[0,220,200]}}}));
  if (action === 'read') return layer ? {features:copy(layer.features), busy:!!layer.brushBusy,
    canUndo:!!layer.history?.length, canRedo:!!layer.future?.length} : null;
  if (action === 'close' || action === 'open') {
    if (layer) {
      state.layers = state.layers.filter(l=>l!==layer);
      if(state.selected?.layer===layer)state.selected=null;
      if(state.picks?.layer===layer)state.picks=null;
      if(state.lastEdited===layer)state.lastEdited=null;
      if(state.drawInto===layer.id)state.drawInto=null;
      state.mergeFirst=null;
      setTool('off');
    }
    if (action === 'open') {
      const image = state.layers.find(l=>l.type==='image');
      if(!image)return false;
      layer = {id:'tls-boundary-draft',type:'shapes',local:'TLS boundary draft',tlsDraft:true,tlsMpp:payload.pixel_um,
        name:'TLS draft · region '+payload.region, image_token:image.token,info:image.info,
        editable:true,regions:true,visible:true,opacity:1,open:true,features:features(payload.features),
        history:[],future:[],hidden:new Set(),classes:[{name:'TLS candidate',color:'#00dcc8'}]};
      state.layers.push(layer);state.drawInto=layer.id;state.lastEdited=layer;
      const selected=layer.features.find(f=>f.properties.key===payload.selectedKey)||layer.features[0];
      state.selected=selected?{layer,id:selected.properties.id}:null;state.picks=null;setTool('off');
    }
    renderShapes();updateToolbar();buildPanel();return true;
  }
  if (!layer) return false;
  if (action === 'freeze') {layer.brushBusy=!!payload.value;layer.editable=!payload.value;updateToolbar();return true;}
  if (layer.brushBusy) throw Error('Wait for the current boundary edit to finish.');
  if (action === 'undo') undoShapes(layer);
  else if (action === 'redo') redoShapes(layer);
  else if (action === 'merge') {state.drawInto=layer.id;setTool('merge');}
  else if (action === 'focus') {
    const selected=layer.features.find(f=>f.properties.key===payload.key);
    state.selected=selected?{layer,id:selected.properties.id}:null;state.picks=null;renderShapes();
  } else if (action === 'replace') {
    pushHistory(layer);layer.features=features(payload.features);state.selected=null;state.picks=null;
    saveShapes(layer);renderShapes();buildPanel();
  }
  return true;
}

function saveShapes(layer) {
  if (layer.preview) { layer.saveError = null; layer.status = tissuePreviewText(layer); shapeStatus(layer); return; }
  if (layer.local) { layer.status = layer.features.length + (layer.tlsDraft ? " TLS boundaries · draft" : " cells, kept in this page until saved."); shapeStatus(layer); return; }
  layer.status = "Saving...";
  layer.saveError = null;
  layer.saveGeneration = (layer.saveGeneration || 0) + 1;
  layer.savePending = true;
  shapeStatus(layer);
  clearTimeout(layer.saveTimer);
  layer.saveTimer = setTimeout(() => flushShapeSave(layer).catch(() => {}), 400);
}

// A save the page stopped waiting for may still have reached the file: ask, and carry on from it when it did
// (instead of refusing the next edit as if another window had changed the file).
async function confirmLateSave(layer) {
  const late = layer.unconfirmedSave;
  if (!late) return;
  try {
    const found = await shapeRequest(layer, '?landed=' + late.id);
    layer.unconfirmedSave = null;
    if (found.landed) {
      layer.revision = found.revision; layer.saveError = null;
      if (late.generation === layer.saveGeneration) layer.savePending = false;
      layer.status = "Saved (the server was slow) " + new Date().toLocaleTimeString() + ".";
    } else if (found.revision === layer.revision) layer.saveError = null;  // it never arrived: send it again
    shapeStatus(layer);
  } catch (_) { /* still unreachable: the next save says so */ }
}

// Only one save per layer is in flight. Later clicks remain instant and save after it.
async function flushShapeSave(layer) {
  clearTimeout(layer.saveTimer);
  while (layer.saveBusy) await layer.saveBusy;
  if (layer.unconfirmedSave && !layer.local) await confirmLateSave(layer);
  if (!layer.savePending || layer.local) return;
  layer.saveError = null;
  const generation = layer.saveGeneration;
  const features = JSON.parse(JSON.stringify(layer.features));
  const saveId = Array.from(crypto.getRandomValues(new Uint8Array(12)), b => b.toString(16).padStart(2, '0')).join('');
  const afterSave = layer.unconfirmedSave?.id;
  layer.saveBusy = (async () => {
    try {
      const result = await shapeRequest(layer, '/save', { features, revision: layer.revision, save_id: saveId, ...(afterSave ? {after_save: afterSave} : {}) });
      layer.unconfirmedSave = null;
      if (result.revision) layer.revision = result.revision;
      if (generation === layer.saveGeneration) {
        layer.savePending = false;
        if (result.features) { layer.features = result.features; renderShapes(); }
      }
      layer.status = layer.savePending ? "Saving…" : "Saved " + new Date().toLocaleTimeString() + ".";
      shapeStatus(layer);
      annotationsChanged(layer);
    } catch (error) {
      if (/did not reply/.test(error.message)) layer.unconfirmedSave = {id: saveId, generation};
      layer.saveError = error.message;
      layer.status = "Save not confirmed: " + error.message;
      shapeStatus(layer); message(layer.status, true); throw error;
    } finally { layer.saveBusy = null; shapeStatus(layer); }
  })();
  shapeStatus(layer);
  await layer.saveBusy;
  if (layer.savePending) await flushShapeSave(layer);
}

async function flushAllShapes() {
  for (const layer of shapeLayers()) await flushShapeSave(layer);
}

// Saving from the footer uses the existing queue and lossless preview save.
// An already-saved layer needs no duplicate write or image reload.
async function savePolygons(layer) {
  if (!layer || layer.local || layer.loadingShapes || layer.shapesLoadError || layer.brushBusy) return false;
  if (layer.preview) return saveTissuePreview(layer);
  if (!layer.editable) return false;
  await flushShapeSave(layer);
  if (!polygonSaveState(layer).saved) return false;
  layer.saveError = null; layer.status = 'Polygons saved.'; shapeStatus(layer);
  return true;
}

function polygonSaveState(layer) {
  return {count:(layer.features || []).length, revision:layer.revision,
    image_token:layer.image_token,
    saved:!!layer.revision && layer.revision !== 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    pending:!!(layer.preview || layer.savePending), preview:!!layer.preview,
    busy:!!(layer.saveBusy || layer.brushBusy || layer.preview?.saving),
    loading:!!layer.loadingShapes, loadError:layer.shapesLoadError || '', error:layer.saveError || '',
    editable:!!(layer.preview ? layer.preview.editable : layer.editable)};
}

window.addEventListener("beforeunload", (event) => {
  if (shapeLayers().some(l => l.savePending || l.saveBusy || l.brushBusy)) { event.preventDefault(); event.returnValue = ""; }
});

function shapeStatus(layer) {
  const node = document.getElementById("status-" + layer.id);
  if (node) node.textContent = layer.status || "";
  document.querySelectorAll('[data-shape-status="' + CSS.escape(layer.id) + '"]').forEach(item => { item.textContent = layer.status || ''; });
  document.querySelectorAll('[data-shape-reload="' + CSS.escape(layer.id) + '"]').forEach(item => { item.disabled = !!(layer.saveBusy || layer.brushBusy || layer.loadingShapes); });
  document.querySelectorAll('[data-shape-retry="' + CSS.escape(layer.id) + '"]').forEach(item => {
    item.hidden = !layer.savePending; item.disabled = !!(layer.saveBusy || layer.brushBusy || layer.loadingShapes);
  });
  state.studio?.shapeStatusChanged?.();
}

function ringArea(ring) {
  let total = 0;
  for (let i = 0; i < ring.length - 1; i++) total += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
  return Math.abs(total) / 2;
}
function featureArea(feature) {
  return polygonsOf(feature.geometry).reduce((sum, rings) =>
    sum + rings.slice(1).reduce((a, ring) => a - ringArea(ring), ringArea(rings[0] || [])), 0);
}
function areaText(pixels) {
  const size = pixelSize();
  if (!size) return Math.round(pixels).toLocaleString() + " px²";
  const square = pixels * size * size;
  return square >= 1e6 ? (square / 1e6).toFixed(2) + " mm²" : Math.round(square).toLocaleString() + " µm²";
}
function featureBounds(feature) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  const visit = (p) => { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); };
  polygonsOf(feature.geometry).forEach((rings) => rings.forEach((ring) => ring.forEach(visit)));
  linesOf(feature.geometry).forEach((line) => line.forEach(visit));
  return { x0: x0, y0: y0, x1: x1, y1: y1 };
}
function contains(feature, x, y) {
  return polygonsOf(feature.geometry).some((rings) => {
    let inside = false;
    rings.forEach((ring) => {
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i], [xj, yj] = ring[j];
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
      }
    });
    return inside;
  });
}

// The smallest editable shape under an image point: a cell wins over the region around it.
function editableShapeAt(point) {
  let best = null;
  shapeLayers().forEach((layer) => {
    if (!canEditPolygons(layer)) return;
    (layer.features || []).forEach((feature) => {
      if (!contains(feature, point.x, point.y)) return;
      const area = featureArea(feature);
      const preferred = artifactPickPriority(layer);
      if (!best || preferred > best.preferred || preferred === best.preferred && area < best.area)
        best = { layer, id:feature.properties.id, area, name:feature.properties.name, preferred };
    });
  });
  return best;
}

function selectShapeAt(point) {
  let best = null;
  shapeLayers().forEach((layer) => {
    if (!layer.visible) return;
    (layer.features || []).forEach((feature) => {
      if (!contains(feature, point.x, point.y)) return;
      const area = featureArea(feature);
      const preferred = artifactPickPriority(layer);
      if (!best || preferred > best.preferred || preferred === best.preferred && area < best.area)
        best = { layer: layer, id: feature.properties.id, area: area, preferred };
    });
  });
  const before = state.selected && state.selected.id;
  state.selected = best ? { layer: best.layer, id: best.id } : null;
  if ((state.selected && state.selected.id) !== before) { renderShapes(); buildPanel(); }
  return best;
}

function zoomTo(feature) {
  const item = baseItem();
  if (!item) return;
  const b = featureBounds(feature);
  const pad = Math.max(b.x1 - b.x0, b.y1 - b.y0) * 0.15 + 20;
  state.viewer.viewport.fitBounds(item.imageToViewportRectangle(b.x0 - pad, b.y0 - pad, b.x1 - b.x0 + 2 * pad, b.y1 - b.y0 + 2 * pad));
}

function drawShapesOnCanvas(context, toCanvas, ratio) {
  shapeLayers().forEach((layer) => {
    if (!layer.visible) return;
    (layer.features || []).forEach((feature) => {
      const color = shapeColor(feature);
      context.beginPath();
      polygonsOf(feature.geometry).forEach((rings) => rings.forEach((ring) => {
        ring.forEach((p, i) => { const [x, y] = toCanvas(p[0], p[1]); if (i) context.lineTo(x, y); else context.moveTo(x, y); });
        context.closePath();
      }));
      context.globalAlpha = layer.opacity * 0.12;
      context.fillStyle = color;
      context.fill("evenodd");
      linesOf(feature.geometry).forEach((line) => line.forEach((p, i) => {
        const [x, y] = toCanvas(p[0], p[1]); if (i) context.lineTo(x, y); else context.moveTo(x, y);
      }));
      context.globalAlpha = layer.opacity;
      context.strokeStyle = color;
      context.lineWidth = 2 * ratio;
      context.stroke();
    });
  });
  context.globalAlpha = 1;
}

async function importShapes(layer, payload) {
  message("Importing...");
  try {
    await flushShapeSave(layer);
    const result = await postJSON("/shapes/" + layer.token + "/import", {...payload, revision:layer.revision});
    pushHistory(layer); state.lastEdited = layer;
    layer.features = result.complete ? result.features : layer.features.concat(result.features || []);
    if (result.revision) layer.revision = result.revision;
    message("Imported " + result.added + " shape(s).");
    renderShapes();
    buildPanel();
  } catch (error) {
    message("Could not import: " + error.message, true);
  }
}

async function downloadShapes(layer) {
  try {
    await flushShapeSave(layer);
    const response = await fetch(api("/shapes/" + layer.token + "/download"), { credentials: "same-origin" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const link = document.createElement("a");
    link.download = match ? match[1] : (layer.name || "shapes") + ".geojson";
    link.href = URL.createObjectURL(await response.blob());
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 5000);
  } catch (error) { message("Could not download: " + error.message, true); }
}

function shapesControls(layer, body) {
  if (layer.editable) {
    const tip = help("Draw with the tools above the image: Rectangle (drag), Polygon (click the corners; " +
      "double-click or Enter to finish), Freehand (drag around an area); Merge (click two touching shapes to join them) and Split " +
      "(drag a line across a shape to cut it). Esc cancels. With a drawing tool on, right-click a " +
      "shape to delete it; Delete removes the selected shape. Ctrl+Z undoes the last drawn or deleted shape, Ctrl+Y redoes it. " +
      (layer.tlsDraft ? "TLS changes are a draft. Use Save edits in Spatial analysis to update the dataset." : "Shapes are saved as you draw; type a name in a shape's row."));
    const classes = layer.classes || [];
    if (classes.length > 1) {
      body.appendChild(el("div", { class: "row field" }, [el("label", { class: "lbl", text: "New shapes" }),
        select(classes.map((c) => [c.name, c.name]), (currentClass(layer) || {}).name, (v) => { layer.default_class = v; }), tip]));
    } else {
      body.appendChild(el("div", { class: "row field" }, [el("span", { class: "muted", text: "Draw with the tools above the image." }), tip]));
    }
  }
  if (layer.local) {
    // Hundreds of cells: a count and Undo, not a row each.
    body.appendChild(el("div", { class: "row" }, [el("span", { class: "muted grow", text: (layer.features || []).length + (layer.tlsDraft ? " TLS boundaries" : " cells") }),
      el("button", { class: "btn small", text: "Undo", title: "Ctrl+Z", onclick: () => undoShapes(layer) }),
      el("button", { class: "btn small", text: "Redo", title: "Ctrl+Y", onclick: () => redoShapes(layer) })]));
    body.appendChild(el("div", { class: "hint", id: "status-" + layer.id, text: layer.status || "" }));
    return;
  }
  const list = el("div", { class: "shape-list" });
  (layer.features || []).forEach((feature) => list.appendChild(shapeRow(layer, feature)));
  if (!(layer.features || []).length) list.appendChild(el("div", { class: "hint", text: "No shapes yet." }));
  body.appendChild(list);
  keepListScroll(list, layer, "shapeScroll");
  const actions = el("div", { class: "row wrap" });
  if (layer.editable) {
    actions.appendChild(el("button", { class: "btn small", text: "Undo", onclick: () => undoShapes(layer) }));
    const sources = el("select", { title: "Add the shapes of a file the project already has" });
    sources.appendChild(el("option", { value: "", text: "Import from the project..." }));
    getJSON("/shapes/" + layer.token + "/sources").then((result) => {
      (result.sources || []).forEach((s) => sources.appendChild(el("option", { value: String(s.id), text: s.name })));
      if (!(result.sources || []).length) sources.disabled = true;
    }).catch(() => { sources.disabled = true; });
    sources.addEventListener("change", () => { if (sources.value !== "") importShapes(layer, { source: Number(sources.value) }); });
    actions.appendChild(sources);
    const file = el("input", { type: "file", accept: ".geojson,.json", hidden: "hidden" });
    file.addEventListener("change", async () => {
      const chosen = file.files && file.files[0];
      if (!chosen) return;
      try {
        const data = JSON.parse(await chosen.text());
        const features = Array.isArray(data) ? data : (data.features || (data.type === "Feature" ? [data] : []));
        await importShapes(layer, { features: features });
      } catch (error) { message("That file is not GeoJSON: " + error.message, true); }
      file.value = "";
    });
    actions.appendChild(file);
    actions.appendChild(el("button", { class: "btn small", text: "Import a file", title: "GeoJSON from QuPath or another tool, in the image's pixel coordinates", onclick: () => file.click() }));
  }
  actions.appendChild(el("button", { class: "btn small", text: "Download", title: "Save these shapes as a GeoJSON file", onclick: () => downloadShapes(layer) }));
  body.appendChild(actions);
  shapeStorageControls(layer, body);
}

function shapeRow(layer, feature) {
  const properties = feature.properties;
  const selected = state.selected && state.selected.id === properties.id;
  const row = el("div", { class: "shape-row" + (selected ? " on" : ""), "data-id": properties.id });
  const classes = layer.classes || [];
  let colour;
  if (layer.editable && classes.length) {
    const current = (properties.classification || {}).name;
    colour = select(classes.map((c) => [c.name, c.name]), current, (v) => {
      const cls = classes.find((c) => c.name === v);
      pushHistory(layer);
      properties.classification = { name: cls.name, color: rgbOf(cls.color) };
      saveShapes(layer); renderShapes();
    });
    colour.classList.add("cls");
  } else {
    colour = el("input", { type: "color", value: shapeColor(feature), title: "Colour" });
    colour.disabled = !layer.editable;
    colour.addEventListener("change", () => {
      pushHistory(layer);
      properties.classification = Object.assign({ name: properties.name }, properties.classification || {}, { color: rgbOf(colour.value) });
      saveShapes(layer); renderShapes();
    });
  }
  const name = el("input", { type: "text", value: properties.name || "", title: "Name" });
  name.disabled = !layer.editable;
  name.addEventListener("change", () => {
    pushHistory(layer);
    properties.name = properties.label = name.value.trim() || properties.name;
    saveShapes(layer);
  });
  [colour, name].forEach((node) => node.addEventListener("click", (e) => e.stopPropagation()));
  row.append(colour, name, el("span", { class: "area", text: areaText(featureArea(feature)) }),
    el("button", { class: "btn small", text: "Go", title: "Zoom to this shape", onclick: (e) => { e.stopPropagation(); row.click(); zoomTo(feature); } }));
  if (layer.editable) {
    row.appendChild(el("button", { class: "btn small", text: "Delete", onclick: (e) => { e.stopPropagation(); deleteShape(layer, properties.id); } }));
  }
  row.addEventListener("click", () => { state.selected = { layer: layer, id: properties.id }; renderShapes(); buildPanel(); if (!layer.local) showRegionCard(layer, feature); });
  return row;
}

// Studio's lists that show annotation names (Export, Subtraction) follow saved edits, names and ticks at once.
function annotationsChanged(layer) {
  document.dispatchEvent(new CustomEvent("spacemap-annotations-changed", { detail: { layer: layer ? layer.id : null } }));
}
