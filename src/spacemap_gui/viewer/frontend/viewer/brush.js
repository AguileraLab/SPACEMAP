/* A brush stroke changes the selected polygon once, on release. Mouse movement
 * only draws the cursor and stroke locally, so image tiles keep loading normally. */
"use strict";

function brushMode(event) {
  const original = event?.originalEvent || event || {};
  if (!["off", "rect", "polygon", "freehand"].includes(state.tool) || state.selectMode || state.selectionEdit || state.drawCallback || state.pointCallback || state.alignMove) return null;
  return original.altKey || state.altDown ? "erase" : original.shiftKey || state.shiftDown ? "add" : null;
}

document.addEventListener("keydown", event => {
  if (event.target.closest?.("input:not([type=checkbox]), textarea, select, [contenteditable]")) return;
  if (event.key === "Alt") { state.altDown = true; if (brushTarget()) event.preventDefault(); }
  if (event.key === "Escape" && state.brushStroke) { state.brushStroke = null; state.brushCancelled = true; clearBrushTrail(); }
  updateBrushCursor();
});
document.addEventListener("keyup", event => { if (event.key === "Alt") state.altDown = false; updateBrushCursor(); });
window.addEventListener("blur", () => { state.altDown = false; state.brushCancelled = !!state.brushStroke; state.brushStroke = null; state.brushPointer = null; clearBrushTrail(); updateBrushCursor(); });

function brushTarget() {
  const picked = state.picks;
  if (picked?.ids.size === 1 && canEditPolygons(picked.layer)) {
    const id = Array.from(picked.ids)[0];
    if (picked.layer.features.some(f => f.properties.id === id)) return {layer:picked.layer,id};
  }
  const selected = state.selected;
  return selected && canEditPolygons(selected.layer) && selected.layer.features.some(f => f.properties.id === selected.id) ? selected : null;
}

function updateBrushCursor() {
  let cursor = $("annotation-brush");
  if (!cursor) { cursor = el("div", {id:"annotation-brush", hidden:true}); $("stage").appendChild(cursor); }
  const mode = state.brushStroke?.mode || brushMode();
  cursor.hidden = !mode || !brushTarget() || !state.brushPointer;
  $("stage").classList.toggle("brushing", !cursor.hidden);
  if (cursor.hidden) return;
  const radius = state.brushStroke?.radius || state.brushRadius || (state.brushRadius = 25);
  const pixels = Math.max(1, radius * Math.abs(state.overlayScale || 1));
  cursor.style.width = cursor.style.height = (pixels * 2) + "px";
  cursor.style.left = state.brushPointer.x + "px"; cursor.style.top = state.brushPointer.y + "px";
  cursor.className = mode;
}
function moveBrushCursor(event) { state.brushPointer = event.position; updateBrushCursor(); }

function brushScroll(event) {
  if (!brushMode(event) || !brushTarget()) return;
  event.preventDefaultAction = true;
  if (state.brushStroke) return; // Keep cursor, painted trail and saved width identical within a stroke.
  state.brushRadius = Math.min(100000, Math.max(0.5, (state.brushRadius || 25) * (event.scroll > 0 ? 1.2 : 1 / 1.2)));
  moveBrushCursor(event);
}

function brushTrail() {
  const stroke = state.brushStroke;
  if (!stroke || !state.svgRoot) return;
  let path = state.brushTrail;
  if (!path) { path = document.createElementNS(SVG_NS, "path"); state.brushTrail = path; }
  path.setAttribute("class", "brush-trail " + stroke.mode);
  path.setAttribute("d", stroke.points.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(""));
  path.setAttribute("stroke-width", stroke.radius * 2);
  state.svgRoot.appendChild(path);
}
function clearBrushTrail() { if (state.brushTrail) { state.brushTrail.remove(); state.brushTrail = null; } }

function beginBrush(event) {
  const mode = brushMode(event);
  if (!mode) return false;
  let selected = brushTarget();
  const point = toImage(event.position);
  if (!selected && point) selected = editableShapeAt(point);
  if (!selected) return false;
  event.preventDefaultAction = true;
  if (state.brushBusy || selected.layer.preview?.saving) { state.brushCancelled = true; message("Finishing the previous edit…"); return true; }
  const feature = selected.layer.features?.find(f => f.properties.id === selected.id);
  if (!feature || !point) return true;
  state.selected = {layer:selected.layer, id:selected.id};
  state.picks = {layer:selected.layer, ids:new Set([selected.id])};
  state.brushCancelled = false;
  drawing.points = null; drawing.start = null; drawing.hover = null;
  renderShapes();
  state.brushStroke = {layer:selected.layer, feature, id:selected.id, mode,
    radius:state.brushRadius || (state.brushRadius = 25), points:[[point.x, point.y]]};
  moveBrushCursor(event); brushTrail();
  return true;
}
function dragBrush(event) {
  const stroke = state.brushStroke;
  if (state.brushCancelled) { event.preventDefaultAction = true; return true; }
  if (!stroke) return false;
  event.preventDefaultAction = true;
  const point = toImage(event.position), last = stroke.points[stroke.points.length - 1];
  if (point && Math.hypot(point.x - last[0], point.y - last[1]) >= Math.max(0.2, stroke.radius / 8)) {
    if (stroke.points.length < 10000) stroke.points.push([point.x, point.y]);
  }
  moveBrushCursor(event); brushTrail();
  return true;
}
async function endBrush(event) {
  if (state.brushCancelled) { state.brushCancelled = false; state.brushClickUntil = performance.now() + 500; event.preventDefaultAction = true; return true; }
  const stroke = state.brushStroke;
  if (!stroke) return false;
  event.preventDefaultAction = true;
  dragBrush(event);
  state.brushStroke = null; state.brushClickUntil = performance.now() + 500;
  const layer = stroke.layer;
  state.brushBusy = true; layer.brushBusy = true;
  shapeStatus(layer);
  message("Applying brush…");
  try {
    await flushShapeSave(layer);
    const before = JSON.stringify(layer.features);
    let result;
    if (layer.local || layer.preview) {
      result = await postJSON("/shapes/edit", {op:"brush", image_token:layer.image_token, source_token:layer.token, preview:!!layer.preview, revision:layer.revision, geometry:stroke.feature.geometry, points:stroke.points, radius:stroke.radius, mode:stroke.mode});
      if (!state.layers.includes(layer) || !layer.features.includes(stroke.feature)) return true;
      const made = replaceShapes(layer, [stroke.feature], result.polygons.map(g => g.coordinates), stroke.feature.properties);
      result.selected = made.map(f => f.properties.id);
    } else {
      result = await postJSON("/shapes/" + layer.token + "/brush", {id:stroke.id, points:stroke.points, radius:stroke.radius, mode:stroke.mode, revision:layer.revision});
      layer.history = layer.history || []; layer.history.push(before); if (layer.history.length > 100) layer.history.shift(); layer.future = [];
      layer.features = result.features; layer.revision = result.revision;
      layer.status = "Saved.";
    }
    if (!state.layers.includes(layer)) return true;
    state.lastEdited = layer;
    state.picks = {layer, ids:new Set(result.selected || [stroke.id])};
    state.selected = state.picks.ids.size ? {layer, id:Array.from(state.picks.ids)[0]} : null;
    state.cardSeq += 1; $("cellcard").hidden = true;
    renderShapes(); buildPanel(); refreshRegionList();
    message((stroke.mode === "add" ? "Painted." : "Erased.") + " Ctrl+Z undoes this stroke.");
  } catch (error) { message(error.message, true); }
  finally { state.brushBusy = false; layer.brushBusy = false; clearBrushTrail(); updateBrushCursor(); buildPanel(); refreshRegionList(); shapeStatus(layer); }
  return true;
}
