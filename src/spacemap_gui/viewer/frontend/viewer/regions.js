/* SPACEMAP viewer, part 6 of 13: regions, polygons to review and name (Studio's pixel classes and marker regions).
 * A region layer is a shapes layer with `regions: true`. Click a region to see its card (its size and the markers
 * high in it); Ctrl-click adds more; Select (on the drawing toolbar) turns a drawn outline into a selection; right-click
 * names the selected regions. Every change can be undone with Ctrl+Z. The viewer's parts share their names and load
 * in order (see index.html). */
"use strict";

const REGION_PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
  "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"];

// Shift held while an outline is drawn in Select adds to the selection.
document.addEventListener("keydown", (event) => { if (event.key === "Shift") state.shiftDown = true; });
document.addEventListener("keyup", (event) => { if (event.key === "Shift") state.shiftDown = false; });
window.addEventListener("blur", () => { state.shiftDown = false; });
// Studio's list of the regions follows the selection and every change.
function refreshRegionList() { if (state.studio && state.studio.refreshLive) state.studio.refreshLive(); }

// Polygon selection and save notifications rebuild both lists. Preserve a label
// being typed, its focus and caret in whichever list the user is editing.
function keepRegionLabelFocus(container) {
  const active=document.activeElement;
  if(!container.contains(active)||!active?.dataset.regionLabel)return ()=>{};
  const id=active.dataset.regionLabel, start=active.selectionStart, end=active.selectionEnd;
  const layer=state.layers.find(l=>l.id===id);if(layer)layer.regionLabelDraft=active.value;
  return ()=>{const input=container.querySelector('[data-region-label="'+CSS.escape(id)+'"]');
    if(input){input.focus({preventScroll:true});input.setSelectionRange(start,end);}};
}

function regionLayers() { return state.layers.filter((l) => l.type === "shapes" && l.regions && l.visible); }
function isPicked(layer, id) { return !!(state.picks && state.picks.layer === layer && state.picks.ids.has(id)); }
function pickedFeatures() {
  const picks = state.picks;
  return picks ? (picks.layer.features || []).filter((f) => picks.ids.has(f.properties.id)) : [];
}
function clearPicks() { if (state.picks) { state.picks = null; renderShapes(); } }

// One optional inspection mode for every annotation type, independent of the
// artifact/training settings. Keep channel colours and existing display limits.
function annotationMarkerOptions() {
  if (!state.annotationMarkers) {
    let saved;
    try { saved = JSON.parse(localStorage.getItem('spacemap-viewer-annotation-markers')); } catch (_) { /* defaults */ }
    state.annotationMarkers = {enabled:saved?.enabled === true,
      count:Number.isInteger(saved?.count) && saved.count > 0 ? saved.count : 3};
  }
  return state.annotationMarkers;
}

function annotationMarkerControls() {
  const options = annotationMarkerOptions();
  const on = el('input', {type:'checkbox', 'aria-label':'Top markers'});
  on.checked = options.enabled;
  const count = el('input', {type:'number', min:'1', step:'1', required:'required',
    value:Number.isFinite(options.count) ? options.count : '', 'aria-label':'Number of top markers'});
  count.disabled = !options.enabled;
  const change = () => {
    options.enabled = on.checked;
    options.count = count.valueAsNumber;
    count.disabled = !on.checked;
    count.setCustomValidity(Number.isInteger(options.count) && options.count > 0 ? '' : 'Enter a whole number of 1 or more.');
    state.annotationMarkerRequest = (state.annotationMarkerRequest || 0) + 1;
    try { localStorage.setItem('spacemap-viewer-annotation-markers', JSON.stringify(options)); } catch (_) { /* private mode */ }
    applyAnnotationMarkers(state.inspectedAnnotation);
  };
  on.addEventListener('change', change);
  count.addEventListener('input', change);
  return el('div', {class:'row annotation-marker-controls'}, [
    el('label', {class:'inline'}, [on, el('span', {text:'Top markers'})]),
    el('label', {class:'inline'}, [el('span', {text:'Number'}), count]),
    help('When on, clicking any annotation shows only its top markers from High here. Number sets how many. Colours and contrast stay as set; new channels get their usual display range. When off, annotation clicks leave channels unchanged. Turning it off keeps the current view.')]);
}

async function applyAnnotationMarkers(inspection) {
  const options = annotationMarkerOptions();
  if (!options.enabled || !Number.isInteger(options.count) || options.count < 1 || !inspection?.rows) return;
  const image = state.layers.find(l => l.type === 'image' && l.token === inspection.layer.image_token && !l.info?.rgb);
  if (!image?.channels) return;
  const request = state.annotationMarkerRequest = (state.annotationMarkerRequest || 0) + 1;
  const current = () => state.inspectedAnnotation === inspection && inspection.seq === state.cardSeq &&
    state.layers.includes(image) && state.layers.includes(inspection.layer) && inspection.layer.visible && options.enabled && request === state.annotationMarkerRequest;
  if (!current()) return;
  const chosen = new Set();
  for (const marker of inspection.rows) {
    const channel = image.channels.find(c => Number.isInteger(marker.index) ? c.index === marker.index && c.name === marker.name : c.name === marker.name);
    if (channel && Number.isFinite(marker.z)) chosen.add(channel.index);
    if (chosen.size >= options.count) break;
  }
  await showRankedMarkerChannels(image, chosen, current);
}

// Shared display operation for annotation and cell inspection. Callers retain
// ownership of their ranking and supply a guard for late async responses.
async function showRankedMarkerChannels(image, chosen, current) {
  if (!chosen.size) return; // Missing/failed rankings must never blank the image.
  const ranges = image.channels.filter(c => chosen.has(c.index) && (c.low === undefined || c.high === undefined)).map(c => ({...c}));
  try {
    await fillLimits(image, ranges);
    if (!current()) return;
    ranges.forEach(value => {
      const channel = image.channels.find(c => c.index === value.index);
      if (channel.low === undefined) channel.low = value.low;
      if (channel.high === undefined) channel.high = value.high;
    });
    image.channels.forEach(c => {c.visible = chosen.has(c.index);});
    image.activeSet = '';
    if (!image.visible) {setVisible(image, true); buildPanel();}
    // Update channel rows in place: keep focus in Number while typing and keep
    // the panel's open groups, marker search and scroll position unchanged.
    image.channels.forEach(channel => {
      const inputs = image.inputs?.[channel.index];
      if (!inputs?.row.isConnected) return;
      inputs.row.querySelector('input[type=checkbox]').checked = channel.visible;
      inputs.row.classList.toggle('off', !channel.visible);
      inputs.low.value = channel.low ?? ''; inputs.high.value = channel.high ?? '';
    });
    image.syncAllMarkers?.();
    const set = document.querySelector('[data-layer-id="' + CSS.escape(image.id) + '"] .sets select');
    if (set) set.value = '';
    remember(image); refreshLayer(image);
  } catch (error) { if (current()) message('Could not show top markers: ' + error.message, true); }
}

function annotationProfile(layer, feature) {
  const cache = state.annotationProfiles || (state.annotationProfiles = new WeakMap());
  let stored = cache.get(feature);
  if (!stored || stored.geometry !== feature.geometry || stored.token !== layer.token) {
    const imported = layer.type === 'annotations';
    const request = postJSON('/' + (imported ? 'annotations' : 'shapes') + '/' + layer.token + '/profile',
      imported ? {index:feature.annotationIndex, stamp:feature.annotationStamp} : {geometry:feature.geometry});
    stored = {geometry:feature.geometry, token:layer.token, request};
    cache.set(feature, stored);
    request.catch(() => {if (cache.get(feature) === stored) cache.delete(feature);});
  }
  return stored.request;
}

async function importedAnnotationClick(point) {
  const layers = state.layers.filter(l => l.type === 'annotations' && l.visible).reverse();
  if (!layers.length) return false;
  const seq = ++state.cardSeq;
  for (const layer of layers) {
    try {
      const found = await getJSON('/annotations/' + layer.token + '/at?x=' + point.x + '&y=' + point.y +
        '&hide=' + encodeURIComponent(Array.from(layer.hidden || []).join('\x1f')));
      if (seq !== state.cardSeq) return true;
      if (!state.layers.includes(layer) || !layer.visible) continue;
      if (found.feature && !layer.hidden?.has(found.feature.properties.name)) {
        showRegionCard(layer, {...found.feature, annotationIndex:found.index, annotationStamp:found.stamp});
        return true;
      }
    } catch (error) {if (seq !== state.cardSeq) return true; message(error.message, true);}
  }
  return false;
}

// ---- Annotations by name (Layers): each polygon belongs to one name row, the same rule as the server's Export
// (studio/annotation_names.py). A name given by hand wins; tissue pieces (tissue_1, tissue_2…) belong to the tissue;
// artifact polygons to their artifact (Blood clots…), as Artifact or Good tissue; otherwise the polygon's class.
const AUTOMATIC_NAMES = /^(Valid_Tissue|Marker_Region|Region \d+|Unnamed|Artifact|Good tissue|Sharp tissue|Good glass)$/;
function annotationPart(layer) { return String(layer.name || '').replace(/ · (Good tissue|Sharp tissue|Good glass)$/, ''); }
function annotationName(layer, feature) {
  const p = feature.properties || {};
  if (layer.artifact_part || layer.training_part) return annotationPart(layer);
  if (p.named && p.name) return String(p.name);
  if (layer.annotation_kind === 'tissue' || layer.working_tissue) return layer.name;
  const cls = (p.classification || {}).name;
  if (cls && !AUTOMATIC_NAMES.test(cls)) return String(cls);
  if (p.name && /^Cluster \d+$/.test(p.name)) return p.name;
  return String(layer.name || 'Annotations').replace(/^Marker regions · /, '');
}
// Artifact layers keep the reviewed artifacts and the training "good" examples apart.
function annotationRole(layer) { return layer.training_part ? 'good' : layer.artifact_part ? 'artifact' : ''; }
// Where a layer's polygons come from: the open image, or an imported file (by its name).
function annotationSource(layer) { return layer.annotation_group === 'imported' ? (layer.source_name || layer.name) : ''; }
function annotationKey(layer, name) { return annotationSource(layer) + '::' + name + '::' + annotationRole(layer); }
function featureShown(layer, feature) { return !(layer.hiddenNames && layer.hiddenNames.has(annotationName(layer, feature))); }

// The project's names given by hand, with their colours (Studio, newest first): every image offers them, so a name
// such as Tumor is typed and coloured once (outputs/studio/annotation_names.json).
function paletteRoute() {
  const session = new URLSearchParams(location.search).get("session");
  return state.studio && session ? "/studio/" + encodeURIComponent(session) + "/annotation_palette" : null;
}
async function refreshNamePalette() {
  const route = paletteRoute();
  if (!route) return state.namePalette || [];
  try { state.namePalette = (await getJSON(route)).names || []; } catch (_) { /* kept as it was */ }
  return state.namePalette;
}
// Keep names ({name: colour}) for the project: shown at once, saved in the background.
function rememberNames(names) {
  const route = paletteRoute(), entries = Object.entries(names || {}).filter(([name, colour]) => name && /^#[0-9a-f]{6}$/i.test(colour));
  if (!route || !entries.length) return;
  const added = new Set(entries.map(([name]) => name.toLowerCase()));
  state.namePalette = [...entries.map(([name, colour]) => ({ name, colour: colour.toLowerCase() })),
    ...(state.namePalette || []).filter((item) => !added.has(item.name.toLowerCase()))];
  postJSON(route, { names: Object.fromEntries(entries) }).then((value) => { state.namePalette = value.names || state.namePalette; }).catch(() => {});
}
function paletteColour(name) {
  const key = String(name).toLowerCase();
  return ((state.namePalette || []).find((item) => item.name.toLowerCase() === key) || {}).colour || null;
}
document.addEventListener("spacemap-annotations-changed", () => { refreshNamePalette(); });

// A name keeps one colour everywhere: the layer's own choice, its colour on this image, the project's colour for it,
// else a colour from the palette by the name's letters.
function nameColour(layer, name) {
  if (layer && layer.name_colours && layer.name_colours[name]) return layer.name_colours[name];
  // The colour this name already has on this image (another layer), so a renamed polygon joins its group's colour.
  for (const other of shapeLayers()) {
    if (other.image_token !== layer?.image_token) continue;
    const found = (other.features || []).find(f => f.properties.named && f.properties.name === name && (f.properties.classification || {}).color);
    if (found) return shapeColor(found);
  }
  const kept = paletteColour(name);
  if (kept) return kept;
  let sum = 0;
  for (const char of String(name)) sum = (sum * 31 + char.charCodeAt(0)) % 9973;
  return REGION_PALETTE[sum % REGION_PALETTE.length];
}
// Names to give polygons: the project's (newest first), then this layer's and the other names on this image.
function regionNames(layer) {
  if (!state.namePalette && !state.namePaletteAsked) { state.namePaletteAsked = true; refreshNamePalette(); }
  const names = new Set(layer.names || []);
  (layer.features || []).forEach((f) => { if (f.properties.name && f.properties.named) names.add(f.properties.name); });
  shapeLayers().forEach((other) => {
    if (other === layer || other.image_token !== layer.image_token || other.artifact_part || other.training_part || other.working_tissue || other.annotation_kind === "tissue") return;
    (other.features || []).forEach((f) => { if (f.properties.named && f.properties.name) names.add(f.properties.name); });
  });
  const kept = (state.namePalette || []).map((item) => item.name);
  const seen = new Set(kept.map((name) => name.toLowerCase()));
  return [...kept, ...Array.from(names).filter((name) => !seen.has(name.toLowerCase())).sort((a, b) => a.localeCompare(b))];
}
function regionTitle(feature) {
  const p = feature.properties;
  return p.named ? p.name : p.cluster !== undefined && p.cluster !== null ? "Cluster " + p.cluster : p.name || "Unnamed";
}

// The smallest visible region under an image point.
function regionAt(point) {
  let best = null;
  regionLayers().forEach((layer) => (layer.features || []).forEach((feature) => {
    if (!featureShown(layer, feature) || !contains(feature, point.x, point.y)) return;
    const area = featureArea(feature);
    const preferred = artifactPickPriority(layer);
    if (!best || preferred > best.preferred || preferred === best.preferred && area < best.area) best = { layer, feature, area, preferred };
  }));
  return best;
}

function afterRegionChange(layer, text) {
  state.lastEdited = layer;
  saveShapes(layer); renderShapes(); buildPanel(); refreshRegionList();
  if (!layer.preview && state.studio && state.studio.regionsChanged) state.studio.regionsChanged(layer);
  if (text) message(text + " Ctrl+Z undoes it.");
}

function labelExamples(layer, ids, role) {
  if (!layer.editable || !ids.size || layer.brushBusy) return;
  pushHistory(layer);
  const name = role === 'positive' ? 'Artifact' : 'Good tissue';
  for (const feature of layer.features) if (ids.has(feature.properties.id)) {
    Object.assign(feature.properties, {example_role:role, name, label:name,
      classification:{name, color:role === 'positive' ? [230,25,75] : [46,204,113]}});
  }
  afterRegionChange(layer, 'Examples labelled ' + name + '.');
}

// Apply a label to all checked regions; clearing a label is a separate explicit action.
function nameRegions(layer, ids, name, {clear=false}={}) {
  const value = String(name || "").trim();
  if (!(layer.editable || layer.preview?.editable) || !ids.size || layer.brushBusy || layer.preview?.saving) return;
  if (!value && !clear) { message('Enter a label for the selected polygons.',true); return; }
  if (!(layer.features||[]).some(f=>ids.has(f.properties.id))) { message('Select the polygons to label first.',true); return; }
  if(value)layer.regionLabelDraft=value;
  if (value) rememberNames({ [value]: nameColour(layer, value) });
  pushHistory(layer);
  (layer.features || []).forEach((f) => {
    if (!ids.has(f.properties.id)) return;
    const p = f.properties;
    if (value) {
      p.name = p.label = value; p.named = true;
      p.classification = { name: value, color: rgbOf(nameColour(layer, value)) };
    } else {
      delete p.named;
      p.name = p.label = p.cluster !== undefined && p.cluster !== null ? "Cluster " + p.cluster : "Region " + (p.number || "");
      p.classification = { name: p.name, color: rgbOf(p.cluster_colour || "#ffd600") };
    }
  });
  afterRegionChange(layer, (value ? "Named " : "Name taken from ") + ids.size + (ids.size === 1 ? " region" : " regions") + (value ? " " + value + "." : "."));
  const inspected=state.inspectedAnnotation;
  if(!$("cellcard").hidden&&inspected?.layer===layer&&ids.has(inspected.feature?.properties.id)){
    const feature=layer.features.find(f=>f.properties.id===inspected.feature.properties.id);
    if(feature)showRegionCard(layer,feature);
  }
}

function deleteRegions(layer, ids) {
  if (!(layer.editable || layer.preview?.editable) || !ids.size || layer.brushBusy || layer.preview?.saving) return;
  pushHistory(layer);
  layer.features = layer.features.filter((f) => !ids.has(f.properties.id));
  if (state.picks && state.picks.layer === layer) state.picks = null;
  if (state.selected && ids.has(state.selected.id)) state.selected = null;
  $("cellcard").hidden = true;
  state.cardSeq += 1;
  afterRegionChange(layer, "Deleted " + ids.size + (ids.size === 1 ? " region." : " regions."));
}

// Click: select a region and show its card; Ctrl/Shift-click adds it to (or takes it from) the selection.
function regionClick(point, additive) {
  const hit = regionAt(point);
  if (!hit) { if (!additive && state.picks) { clearPicks(); buildPanel(); refreshRegionList(); } return false; }
  const id = hit.feature.properties.id;
  if (additive && state.picks && state.picks.layer === hit.layer) {
    if (state.picks.ids.has(id)) state.picks.ids.delete(id); else state.picks.ids.add(id);
  } else state.picks = { layer: hit.layer, ids: new Set([id]) };
  state.selected = { layer: hit.layer, id };
  renderShapes(); buildPanel(); refreshRegionList();
  showRegionCard(hit.layer, hit.feature);
  return true;
}

// Select: every region whose middle lies inside the outline drawn (rectangle, polygon or freehand).
function regionCentre(feature) {
  const rings = polygonsOf(feature.geometry)[0];
  if (!rings || !rings[0] || !rings[0].length) { const b = featureBounds(feature); return [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2]; }
  let x = 0, y = 0, a = 0;
  const ring = rings[0];
  for (let i = 0; i < ring.length - 1; i++) {
    const cross = ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
    a += cross; x += (ring[i][0] + ring[i + 1][0]) * cross; y += (ring[i][1] + ring[i + 1][1]) * cross;
  }
  if (Math.abs(a) < 1e-9) { const b = featureBounds(feature); return [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2]; }
  return [x / (3 * a), y / (3 * a)];
}
function selectInside(points, additive) {
  const outline = { geometry: { type: "Polygon", coordinates: [points.map((p) => [p.x, p.y]).concat([[points[0].x, points[0].y]])] } };
  let best = null;
  regionLayers().forEach((layer) => {
    const ids = (layer.features || []).filter((f) => { const [x, y] = regionCentre(f); return contains(outline, x, y); }).map((f) => f.properties.id);
    const preferred = layer.id === state.drawInto;  // the current stage's layer first
    if (ids.length && (!best || preferred && !best.preferred || preferred === best.preferred && ids.length > best.ids.length)) best = { layer, ids, preferred };
  });
  if (!best) { message("No region inside that outline."); return; }
  if (additive && state.picks && state.picks.layer === best.layer) best.ids.forEach((id) => state.picks.ids.add(id));
  else state.picks = { layer: best.layer, ids: new Set(best.ids) };
  renderShapes(); buildPanel(); refreshRegionList();
  message(state.picks.ids.size + (state.picks.ids.size === 1 ? " region" : " regions") + " selected: right-click one to name them.");
}

// The regions a right click acts on: the selection when the clicked region is in it, else the clicked one.
function regionMenuItems(point, act) {
  const hit = point && regionAt(point);
  if (!hit || !canEditPolygons(hit.layer)) return [];
  const layer = hit.layer, id = hit.feature.properties.id;
  const ids = isPicked(layer, id) ? new Set(state.picks.ids) : new Set([id]);
  const what = ids.size === 1 ? regionTitle(hit.feature) : ids.size + " regions";
  if (layer.training_part || layer.tlsDraft) return [
    el('div', {class:'ctx-title', text:what}),
    act('Zoom to it', () => zoomTo(hit.feature)),
    act('Delete selected', () => deleteRegions(layer, ids)), el('hr')];
  const names = regionNames(layer);
  const typed = () => {
    const value = window.prompt("Name for " + what + ":", "");
    if (value && value.trim()) nameRegions(layer, ids, value.trim());
  };
  refreshNamePalette();  // names given on another image since, for the next menu
  const named = (name) => {
    const button = act(name, () => nameRegions(layer, ids, name));
    button.prepend(el("span", { class: "ctx-swatch", style: "background:" + nameColour(layer, name), "aria-hidden": "true" }));
    button.title = "Name " + what + " " + name;
    return button;
  };
  return [
    el("div", { class: "ctx-title", text: "Name " + what + " as" }),
    ...names.slice(0, 20).map(named),
    act("New name…", typed),
    act("Take the name away", () => nameRegions(layer, ids, "", {clear:true})),
    act("Zoom to " + (ids.size === 1 ? "it" : "them"), () => zoomTo({ geometry: { type: "MultiPolygon", coordinates:
      (layer.features || []).filter((f) => ids.has(f.properties.id)).flatMap((f) => polygonsOf(f.geometry)) } })),
    act("Delete " + (ids.size === 1 ? "it" : "them"), () => deleteRegions(layer, ids)),
    el("hr"),
  ];
}

// The region's card: its name, size and the markers that stand out in it (against the tissue).
async function showRegionCard(layer, feature) {
  const card = $("cellcard");
  const seq = ++state.cardSeq;
  const inspection = state.inspectedAnnotation = {layer, feature, seq};
  state.inspectedCell = null; state.cellInspectionSelection = null;
  releaseSplitMarkerSelection();
  state.cellInspectEpoch = (state.cellInspectEpoch || 0) + 1;
  state.cellOutline = null; drawCellOutline();
  card.hidden = false; card.innerHTML = "";
  const p = feature.properties;
  const close = el("button", { class: "btn small", text: "Close", onclick: () => { card.hidden = true; state.cardSeq += 1; } });
  card.appendChild(el("h4", {}, [el("span", { text: regionTitle(feature) }), close]));
  const facts = [["Area", layer.tlsDraft ? (featureArea(feature)*layer.tlsMpp**2).toLocaleString(undefined,{maximumFractionDigits:1})+' µm²' : areaText(featureArea(feature))]];
  if (p.cluster !== undefined && p.cluster !== null) facts.push(["Cluster", String(p.cluster)]);
  if (p.named) facts.push(["Name", p.name]);
  card.appendChild(el("div", { class: "region-facts" }, facts.map(([k, v]) => el("div", {}, [el("span", { class: "muted", text: k + " " }), el("b", { text: v })]))));
  if(layer.tlsDraft){
    card.append(el('p',{class:'muted',text:'TLS boundary draft · use Save edits in Spatial analysis.'}));
    return;
  }
  if (layer.training_part) {
    card.append(el('div',{class:'row'},[el('span',{class:'muted',text:layer.training_part==='lint'?'Good glass example':'Good tissue example'}),el('button',{class:'btn small',text:'Zoom',onclick:()=>zoomTo(feature)})]));
  } else if (layer.editable) {
    const list = "region-names-" + layer.id;
    const input = el("input", { type: "text", list, placeholder: "Polygon label…", value: p.named ? p.name : "", "aria-label": "Region name", maxlength: "60" });
    const options = el("datalist", { id: list }, regionNames(layer).map((name) => el("option", { value: name })));
    const apply = () => nameRegions(layer, new Set([p.id]), input.value);
    input.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); apply(); } });
    card.appendChild(el("div", { class: "row" }, [input, options, el("button", { class: "btn small", text: "Apply label", onclick: apply }),
      el("button", { class: "btn small ghost", text: "Zoom", onclick: () => zoomTo(feature) })]));
  }
  const holder = el("div", { class: "region-profile" }, [el("p", { class: "muted", text: "Reading its markers…" })]);
  card.appendChild(holder);
  try {
    const found = await annotationProfile(layer, feature);
    if (seq !== state.cardSeq) return;
    const rows = found.markers || [];
    inspection.rows = rows;
    applyAnnotationMarkers(inspection);
    const top = rows.slice(0, 10), low = rows.slice(-3).reverse().filter((m) => m.z < 0 && !top.includes(m));
    const limit = Math.max(3, ...rows.map((m) => Math.abs(m.z)));
    const bar = (m) => el("div", { class: "region-marker" + (m.z < 0 ? " low" : "") }, [
      el("span", { class: "grow", text: m.name, title: m.name }),
      el("span", { class: "region-bar" }, [el("i", { style: "width:" + Math.min(100, (100 * Math.abs(m.z)) / limit).toFixed(0) + "%" })]),
      el("span", { class: "mono", text: (m.z >= 0 ? "+" : "") + m.z.toFixed(1), title: "Robust z: how many spreads above the " + found.reference + "’s middle level" }),
      el("span", { class: "mono muted", text: "×" + m.fold.toFixed(1), title: "Mean here over the " + found.reference + "’s mean" })]);
    holder.replaceChildren(
      el("div", { class: "row" }, [el("b", { text: "High here" }), el("span", { class: "grow" }),
        help("Each marker’s mean inside this region against the " + found.reference + ": z is how many spreads (robust SD) above the " +
          found.reference + "’s middle level, × is the mean here over the mean there. Region measured at " + found.microns + " µm/px; comparison at " + found.reference_microns + " µm/px.")]),
      ...top.map(bar),
      ...(low.length ? [el("div", { class: "muted", text: "Low here" })] : []), ...low.map(bar));
  } catch (error) { if (seq === state.cardSeq) holder.replaceChildren(el("p", { class: "hint", text: error.message })); }
}

// A region layer's list in Studio: sort by place or size, name the selected, drop the small ones.
function regionsControls(layer, body) {
  if (layer.artifact_part && body.dataset.layerControls === "right") state.studio?.renderArtifactControls?.(layer, body);
  const onlyClass=body.id==='studio-tissue-polygons' ? layer.working_class : '';
  const onlyName = body.dataset.onlyName;  // a name row in Layers lists only its own polygons
  const features = (layer.features || []).filter(f=>(!onlyClass || f.properties.classification?.name===onlyClass) &&
    (onlyName === undefined || annotationName(layer, f) === onlyName));
  const sort = layer.sortBy || "place";
  const band = Math.max(1, ...features.map((f) => { const b = featureBounds(f); return b.y1 - b.y0; })) || 1;
  const ordered = features.slice().sort(sort === "size" ? (a, b) => featureArea(b) - featureArea(a) : (a, b) => {
    const [ax, ay] = regionCentre(a), [bx, by] = regionCentre(b);
    const rowA = Math.floor(ay / band), rowB = Math.floor(by / band);
    return rowA - rowB || ax - bx;
  });
  const existing = new Set(features.map(f => f.properties.id));
  const picked = state.picks && state.picks.layer === layer ? new Set(Array.from(state.picks.ids).filter(id => existing.has(id))) : new Set();
  const pick = ids => {
    state.picks = {layer, ids};
    if (state.selected?.layer === layer && !ids.has(state.selected.id)) state.selected = null;
    renderShapes(); buildPanel(); refreshRegionList();
  };
  const all = el('input', {type:'checkbox', 'aria-label':'Select all polygons in ' + layer.name});
  all.checked = !!features.length && picked.size === features.length;
  all.indeterminate = picked.size > 0 && picked.size < features.length;
  all.disabled = !features.length || !!layer.brushBusy || !!layer.preview?.saving;
  all.addEventListener('change', () => pick(all.checked ? new Set(existing) : new Set()));
  const remove = el('button', {class:'btn small', text:'Delete selected', disabled:(layer.editable || layer.preview?.editable) && picked.size && !layer.brushBusy && !layer.preview?.saving ? null : '',
    title:'Delete the selected polygons. Ctrl+Z brings them back.', onclick:() => deleteRegions(layer, new Set(picked))});
  const convert=(layer.artifact_part||layer.training_part)?el('button', {class:'btn small',
    text:layer.training_part?'Mark as artifact':layer.artifact_part==='blur'?'Mark as sharp tissue':layer.artifact_part==='lint'?'Mark as good glass':'Mark as good tissue',
    disabled:(layer.editable||layer.preview?.editable)&&picked.size&&!state.brushBusy&&!layer.preview?.saving?null:'',
    title:'Move every selected polygon to the other training class. Ctrl+Z brings them back.',
    onclick:()=>convertArtifactSelection(layer,Array.from(picked),layer.training_part?'positive':'negative')}):null;
  body.appendChild(el('div', {class:'row wrap region-selection'}, [el('label', {class:'inline'}, [all, el('span', {text:'All polygons'})]),
    el('span', {class:'grow muted', text:picked.size + ' selected'}), ...(convert?[convert]:[]), remove]));
  const order = select([["place", "By place (top left first)"], ["size", "By size (largest first)"]], sort, (v) => { layer.sortBy = v; buildPanel(); refreshRegionList(); });
  const list = "region-names-list-" + layer.id + '-' + newId();
  const nameBox = el("input", { type: "text", list, value:layer.regionLabelDraft||'', placeholder: "Label selected polygons…", maxlength: "60", "aria-label": "Name the selected regions", 'data-region-label':layer.id });
  const names = el("datalist", { id: list }, regionNames(layer).map((name) => el("option", { value: name })));
  const nameButton = el("button", { class: "btn small", text: "Apply label", title:'Apply this label to every checked polygon. Names save automatically; Ctrl+Z undoes.', onclick: () => nameRegions(layer, new Set(picked), nameBox.value) });
  const syncName=()=>{nameButton.disabled=!picked.size||!nameBox.value.trim()||!!layer.brushBusy||!!layer.preview?.saving;};
  nameBox.addEventListener('input',()=>{layer.regionLabelDraft=nameBox.value;syncName();});
  nameBox.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();nameButton.click();}});syncName();
  const smallest = el("input", { type: "number", min: "0", step: "100", value: String(layer.minArea || 0), "aria-label": "Remove regions smaller than (µm²)", title: "µm²" });
  const size = pixelSize();
  const drop = el("button", { class: "btn small", text: "Remove smaller", title: "Delete every region smaller than this (µm²); Ctrl+Z brings them back", onclick: () => {
    layer.minArea = Number(smallest.value) || 0;
    const limit = size ? layer.minArea / (size * size) : layer.minArea;
    deleteRegions(layer, new Set(features.filter((f) => featureArea(f) < limit).map((f) => f.properties.id)));
  } });
  body.appendChild(el("div", { class: "row wrap" }, [order, el("span", { class: "muted grow", text: features.length + " regions" + (picked.size ? " · " + picked.size + " selected" : "") }),
    el("button", { class: "btn small", text: "Undo", title: "Ctrl+Z", onclick: () => { state.lastEdited = layer; undoShapes(layer); } })]));
  if (layer.editable) {
    if (!layer.training_part && !layer.tlsDraft) body.appendChild(el("div", { class: "row wrap" }, [nameBox, names, nameButton, el("span", { class: "grow" }), smallest, el("span", { class: "muted", text: "µm²" }), drop]));
  }
  const rows = el("div", { class: "shape-list region-list" });
  const shown = ordered.slice(0, layer.showAll ? ordered.length : 300);
  shown.forEach((feature) => {
    const p = feature.properties;
    const row = el("div", { class: "shape-row" + (picked.has(p.id) ? " on" : ""), "data-id": p.id });
    const tick = el('input', {type:'checkbox', 'aria-label':'Select ' + regionTitle(feature), 'data-region-id':p.id});
    tick.checked = picked.has(p.id); tick.disabled = !!layer.brushBusy || !!layer.preview?.saving;
    tick.addEventListener('click', event => event.stopPropagation());
    tick.addEventListener('change', () => { const ids = new Set(picked); if (tick.checked) ids.add(p.id); else ids.delete(p.id); pick(ids); });
    row.append(tick, el("span", { class: "swatch", style: "background:" + shapeColor(feature) }),
      el("span", { class: "grow region-name", text: regionTitle(feature), title: regionTitle(feature) }),
      el("span", { class: "area", text: layer.tlsDraft ? (featureArea(feature)*layer.tlsMpp**2).toLocaleString(undefined,{maximumFractionDigits:1})+' µm²' : areaText(featureArea(feature)) }),
      el("button", { class: "btn small", text: "Go", title: "Zoom to it", onclick: (event) => { event.stopPropagation(); row.click(); } }));
    row.addEventListener("click", (event) => {
      const additive = event.ctrlKey || event.metaKey || event.shiftKey;
      if (additive && state.picks && state.picks.layer === layer) { if (state.picks.ids.has(p.id)) state.picks.ids.delete(p.id); else state.picks.ids.add(p.id); }
      else { state.picks = { layer, ids: new Set([p.id]) }; zoomTo(feature); }
      state.selected = { layer, id: p.id };
      renderShapes(); buildPanel(); refreshRegionList();
      showRegionCard(layer, feature);
    });
    rows.appendChild(row);
  });
  if (ordered.length > shown.length) rows.appendChild(el("button", { class: "btn small ghost", text: "Show all " + ordered.length, onclick: () => { layer.showAll = true; refreshRegionList(); } }));
  if (!features.length) rows.appendChild(el("div", { class: "hint", text: "No regions yet." }));
  body.appendChild(rows);
  keepListScroll(rows, layer, "regionScroll");
  shapeStorageControls(layer, body);
  body.appendChild(el("button", {class:"btn small", text:"Download", title:"Download these polygons as GeoJSON", onclick:()=>downloadShapes(layer)}));
  if (layer.annotation_kind) annotationHistory(layer, body);
}

function annotationHistory(layer, body) {
  const section = el("details", {class:"annotation-history-box"});
  section.open = !!layer.historyOpen;
  section.appendChild(el("summary", {text:"History"}));
  const list = el("div", {class:"annotation-history"});
  const note = el("p", {class:"muted", text:"Earlier versions stay available after restoring."});
  const draw = result => {
    if (!list.isConnected) return;
    const entries = result.entries || [];
    const rows = entries.map(entry => ({...entry, version:entry.after}));
    if (entries.length) rows.push({version:entries[entries.length - 1].before, label:"Before the first change", when:""});
    list.replaceChildren(...rows.map(entry => {
      const restore = el("button", {class:"btn small", text:entry.version === layer.revision ? "Current" : "Restore", disabled:entry.version === layer.revision ? "" : null});
      restore.addEventListener("click", async () => {
        restore.disabled = true;
        try {
          await flushShapeSave(layer);
          const before = JSON.stringify(layer.preview ? layer.preview.features : layer.features);
          const result = await postJSON("/shapes/" + layer.token + "/history", {version:entry.version, revision:layer.revision});
          clearShapePreview(layer, false);
          layer.history = layer.history || []; layer.history.push(before); layer.future = [];
          layer.features = result.features; layer.revision = result.revision; state.lastEdited = layer;
          state.selected = null; state.picks = null; state.cardSeq += 1; $("cellcard").hidden = true;
          layer.status = "Restored. The previous version is in History.";
          renderShapes(); buildPanel(); refreshRegionList();
        } catch (error) { note.textContent = error.message; restore.disabled = false; }
      });
      return el("div", {class:"row"}, [el("span", {class:"grow", text:entry.label, title:entry.when ? new Date(entry.when).toLocaleString() : ""}), restore]);
    }));
    if (!rows.length) list.appendChild(el("p", {class:"muted", text:"Versions appear after the first change."}));
  };
  const load = async () => {
    try {
      if (layer.historyCache?.revision === layer.revision) { draw(layer.historyCache); return; }
      const found = await getJSON("/shapes/" + layer.token + "/history");
      layer.historyCache = found; draw(found);
    } catch (error) { note.textContent = error.message; }
  };
  section.append(list, note);
  section.addEventListener("toggle", () => { layer.historyOpen = section.open; if (section.open) load(); });
  body.appendChild(section);
  if (section.open) load();
}


// A polygon list is drawn again after every click (Go, a tick, a label): it keeps where it was scrolled to, and only
// moves to show the selected polygon when that one is out of sight (picked on the image, say).
function keepListScroll(list, layer, key) {
  list.addEventListener("scroll", () => { layer[key] = list.scrollTop; });
  const restore = () => {
    if (!list.isConnected) return;
    if (layer[key]) list.scrollTop = layer[key];
    const chosen = state.selected && state.selected.layer === layer && list.querySelector('[data-id="' + CSS.escape(String(state.selected.id)) + '"]');
    if (chosen) {
      const box = list.getBoundingClientRect(), row = chosen.getBoundingClientRect();
      if (row.top < box.top || row.bottom > box.bottom) { chosen.scrollIntoView({ block: "nearest" }); layer[key] = list.scrollTop; }
    }
  };
  restore();
  requestAnimationFrame(restore);  // the list is put on the page after this function returns
}
