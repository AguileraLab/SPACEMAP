// Studio's Annotations in Layers: Working annotations and Imported GeoJSON, then one row per
// annotation name (tissue, Blood clots, Epithelia, Tumor…), whatever layer file its polygons are kept in. A row shows or
// hides its polygons, sets their colour, says where they came from, ticks them for Export, and lists them (select, Go,
// label, delete) with the layer's own tools. Artifacts split into Bad tissue and Good tissue (training examples).

const ROLE_NAMES = { artifact: "Bad tissue", good: "Good tissue" };

function annotationExportState() {
  const image = state.layers.find((l) => l.type === "image");
  const token = image && image.token;
  if (!token) return null;
  if (!state.annotationExport || state.annotationExport.token !== token) {
    state.annotationExport = { token, choices: {}, loading: true };
    const session = new URLSearchParams(location.search).get("session");
    getJSON("/studio/" + encodeURIComponent(session) + "/annotation_names?image_token=" + encodeURIComponent(token))
      .then((found) => { if (state.annotationExport.token === token) { state.annotationExport = { token, choices: found.choices || {}, rules: found.rules || {} }; buildPanel(); } })
      .catch(() => { if (state.annotationExport.token === token) state.annotationExport.loading = false; });
  }
  return state.annotationExport;
}

// A name row is ticked when the project's export rule for it (<origin>:<name>, the same for every image) says Export.
function exportChosen(key, role) {
  const saved = annotationExportState();
  if (saved && key in saved.choices) return !!saved.choices[key];
  if (!saved || role === "good") return false;  // training examples have no rule
  const [source, name] = key.split("::");
  return !!((saved.rules || {})[(source ? "imported:" : "local:") + name] || {}).export;  // a name made since the list was read
}

async function setExportChosen(key, value) {
  const saved = annotationExportState();
  if (!saved) return;
  saved.choices[key] = value;
  const session = new URLSearchParams(location.search).get("session");
  try { await postJSON("/studio/" + encodeURIComponent(session) + "/annotation_names", { image_token: saved.token, choices: { [key]: value } }); annotationsChanged(null); }
  catch (error) { message(error.message, true); }
}

function namesOf(layer) {
  const names = new Set((layer.features || []).map((f) => annotationName(layer, f)));
  return names;
}

function nameShown(layer, name) { return layer.visible && !(layer.hiddenNames && layer.hiddenNames.has(name)); }

// Show or hide one name's polygons; showing it on a hidden layer shows that name only.
function setNameShown(layer, name, on, redraw = true) {
  const all = namesOf(layer);
  if (on) {
    if (!layer.visible) { all.delete(name); layer.hiddenNames = all; setVisible(layer, true); }
    else if (layer.hiddenNames) layer.hiddenNames.delete(name);
  } else {
    layer.hiddenNames = layer.hiddenNames || new Set();
    layer.hiddenNames.add(name);
    if (!all.size || Array.from(all).every((n) => layer.hiddenNames.has(n))) { layer.hiddenNames = null; setVisible(layer, false); }
  }
  if (redraw) { saveView(); renderShapes(); syncAnnotationVisibility(); }
}

function annotationVisibility(layers, title) {
  const show = el('input', {type:'checkbox', 'aria-label':'Show all annotations in ' + title,
    title:'Show or hide all annotations','data-annotation-visibility':''});
  show.refreshVisibility=()=>{
  const flags = layers.flatMap(layer => layer.review_inline ? (layer.features||[]).map(f=>layer.visible&&featureShown(layer,f)) : layer.type === 'shapes' ?
    Array.from(namesOf(layer), name => nameShown(layer, name)) : layer.annotationLegend?.items?.map(item=>layer.visible&&!layer.hidden.has(item.value)) || [layer.visible]);
  show.checked = flags.length > 0 && flags.every(Boolean) && !layers.some(layer => layer.type === 'annotations' && layer.hidden?.size);
  show.indeterminate = flags.some(Boolean) && !show.checked;
  };show.refreshVisibility();
  // Read-only GeoJSON classes are loaded lazily, unlike editable polygons.
  if (layers.length === 1 && layers[0].type === 'annotations') {
    const layer = layers[0];
    annotationLegend(layer).then(legend => {
      if (!show.isConnected) return;
      show.refreshVisibility();
    }).catch(() => {}); // the layer's legend reports its own read errors
  }
  show.disabled = !layers.length;
  show.addEventListener('click', event => event.stopPropagation());
  show.addEventListener('change', () => {
    const on = show.checked;
    layers.forEach(layer => {
      layer.hiddenNames = null;
      if(on&&layer.review_inline)layer.hiddenReviewIds=new Set();
      if (on && layer.type === 'annotations') layer.hidden = new Set();
      setVisible(layer, on);
      if (on && layer.type === 'annotations') refreshLayer(layer);
    });
    saveView(); renderShapes(); syncAnnotationVisibility();
  });
  return show;
}

// Visibility is a display update, not a reason to replace thousands of list
// rows or reload their metadata. Keep focus, disclosure and scroll intact.
function syncAnnotationVisibility() {
  document.querySelectorAll('[data-annotation-visibility]').forEach(node=>node.refreshVisibility?.());
  const count=document.querySelector('[data-layer-group="annotations"] > summary .group-count');
  const layers=state.layers.filter(l=>['shapes','annotations'].includes(l.type));
  if(count)count.textContent=layers.filter(l=>l.visible).length+' shown / '+layers.length;
}
function annotationDisplayChanged(layer) {
  annotationOutline(layer);
  state.versions[layer.id]=(state.versions[layer.id]||0)+1;
  refreshLayer(layer);saveView();syncAnnotationVisibility();
}

// Every polygon of a name takes the colour (saved with its layer; Export writes it for QuPath).
function setNameColour(parts, name, colour) {
  let unsaved = false;
  parts.forEach(({ layer }) => {
    (layer.features || []).forEach((f) => {
      if (annotationName(layer, f) !== name) return;
      const cls = f.properties.classification || {};
      f.properties.classification = { ...cls, name: cls.name || name, color: rgbOf(colour) };
      f.properties.colour = colour;  // kept even where the class colour is fixed (tissue)
    });
    layer.name_colours = { ...(layer.name_colours || {}), [name]: colour };
    if (layer.editable) saveShapes(layer); else unsaved = true;
  });
  // A name given by hand keeps this colour on every image of the project (not tissue or artifact names).
  if (!parts.some(({ layer }) => annotationRole(layer) || layer.working_tissue || layer.annotation_kind === "tissue")) rememberNames({ [name]: colour });
  renderShapes(); buildPanel();
  message(name + " is now " + colour + (unsaved ? " on the image (a read-only layer keeps its own colours in the file)." : "."));
}

// A copy of a name's polygons as a new layer of this image, under a new name, to rename further.
async function duplicateName(parts, name) {
  const image = state.layers.find((l) => l.type === "image");
  const next = window.prompt("Name of the copy of " + name + ":", name + "_copy");
  if (!image || !next || !next.trim()) return;
  const features = parts.flatMap(({ layer }) => (layer.features || []).filter((f) => annotationName(layer, f) === name)).map((f) => {
    const properties = { ...f.properties, name: next.trim(), label: next.trim(), named: true, copied_from: name };
    properties.classification = { name: next.trim(), color: (f.properties.classification || {}).color || rgbOf(nameColour(parts[0].layer, name)) };
    delete properties.id;
    return { type: "Feature", geometry: f.geometry, properties };
  });
  if (!features.length) { message("There are no polygons to copy.", true); return; }
  const session = new URLSearchParams(location.search).get("session");
  try {
    const found = await postJSON("/studio/" + encodeURIComponent(session) + "/annotation_sources",
      { action: "duplicate", image_token: image.token, name: next.trim(), from: name, features });
    if (!state.layers.includes(image)) return;
    await addLayerToView({ ...found.layer, visible: true });
    buildPanel();
    message(features.length + " polygons copied as " + next.trim() + ". Select some and right-click to rename them.");
  } catch (error) { message(error.message, true); }
}

function studioPost(path, body) {
  const session = new URLSearchParams(location.search).get("session");
  return postJSON("/studio/" + encodeURIComponent(session) + path, body);
}

// Layers Studio made or imported can leave the project whole; tissue and artifact layers keep their (empty) place.
function removableLayer(layer) {
  return !layer.artifact_part && !layer.training_part && !layer.working_tissue && layer.annotation_kind !== "tissue" &&
    ["imported", "regions", "classes"].includes(layer.annotation_group);
}

// Delete a name: its polygons (Ctrl+Z and History bring them back); a removable layer holding only this name leaves
// the project (its file is kept in outputs/studio/deleted).
async function deleteName(parts, name) {
  const image = state.layers.find((l) => l.type === "image");
  const total = parts.reduce((sum, p) => sum + p.count, 0);
  const whole = parts.filter(({ layer }) => removableLayer(layer) && (layer.features || []).every((f) => annotationName(layer, f) === name));
  const blocked = parts.filter(({ layer }) => !layer.editable && !whole.includes(parts.find((p) => p.layer === layer)));
  if (blocked.length) { message("This project is open for viewing only, so " + name + " cannot be deleted.", true); return; }
  if (!window.confirm("Delete " + name + " (" + total + " polygon" + (total === 1 ? "" : "s") + ")?" +
      (whole.length ? " Its layer leaves the project; the file is kept in outputs/studio/deleted." : " Ctrl+Z or History brings them back."))) return;
  for (const { layer } of parts) {
    if (whole.some((p) => p.layer === layer)) {
      try {
        await studioPost("/annotation_sources", { action: "remove_layer", image_token: image.token, layer_token: layer.token });
        removeLayer(layer, true);
      } catch (error) { message(error.message, true); return; }
    } else {
      deleteRegions(layer, new Set((layer.features || []).filter((f) => annotationName(layer, f) === name).map((f) => f.properties.id)));
    }
  }
  buildPanel();
  message(name + " deleted.");
}

// A file read as it is (zones, classic outputs, a GeoJSON added from disk): off this image's list; the file stays.
async function hideFile(layer) {
  const image = state.layers.find((l) => l.type === "image");
  if (!window.confirm("Take " + layer.name + " off this image's annotations? The file itself stays where it is.")) return;
  try {
    await studioPost("/annotation_sources", { action: "hide_file", image_token: image.token, layer_token: layer.token });
    removeLayer(layer, true); buildPanel();
  } catch (error) { message(error.message, true); }
}

function annotationOrigin(layer) {
  if (layer.origin) return layer.origin;
  if (layer.annotation_kind === "tissue" || layer.working_tissue) return "Tissue";
  if (layer.training_part) return "Artifacts: training examples";
  if (layer.artifact_part) return "Artifacts";
  if (layer.annotation_group === "imported") return "Imported from " + (layer.source_name || layer.name);
  return { classes: "Classification", regions: "Marker regions", zones: "Zones" }[layer.annotation_group] || "Drawn";
}

function nameOrder(name, parts) {
  const layer = parts[0].layer;
  return (layer.annotation_kind === "tissue" || layer.working_tissue ? "0" : layer.artifact_part || layer.training_part ? "1" : "2") + name.toLowerCase();
}

function canGroupAnnotation(layer) {
  return !!(state.studio||state.reviewer) && layer.regions && !isTrainingCellLayer(layer) && !layer.artifact_part && !layer.training_part;
}
function annotationRowKey(layer, name) {
  return (annotationSource(layer) ? annotationFileKey(layer) : '') + '::' + name;
}
function groupAnnotationDisplay(layer, feature) {
  // Opening the component list changes the controls, not the renderer. Keep
  // exact compound paths and a separate selected outline while the list is open.
  return canGroupAnnotation(layer);
}

function partRows(name, parts, open, source) {
  const holder = el("div", { class: "annotation-name-parts" });
  const byRole = parts.some((p) => annotationRole(p.layer));
  parts.forEach(({ layer, count }) => {
    const role = annotationRole(layer);
    const box = el(byRole ? "details" : "div", { class: "annotation-name-part", "data-layer-id": layer.id });
    const roleKey = source + '::' + layer.id + '::' + name;
    const roleState = state.annotationRolesOpen || (state.annotationRolesOpen = {});
    if (byRole) box.open = roleState[roleKey] ?? layer.id === state.drawInto;
    if (byRole) {
      const key = annotationKey(layer, name);
      const show = el("input", { type: "checkbox", "aria-label": "Show " + name + " " + ROLE_NAMES[role], 'data-annotation-visibility':'' });
      show.refreshVisibility=()=>{show.checked=nameShown(layer,name);};show.refreshVisibility();
      show.addEventListener("change", () => setNameShown(layer, name, show.checked));
      const tick = el("input", { type: "checkbox", "aria-label": "Export " + name + " " + ROLE_NAMES[role] });
      tick.checked = exportChosen(key, role);
      tick.addEventListener("change", () => setExportChosen(key, tick.checked));
      const own = (layer.features || []).find((f) => annotationName(layer, f) === name);
      const swatch = el("input", { type: "color", value: own ? shapeColor(own) : role === "good" ? "#2ecc71" : "#e6194b",
        "aria-label": "Colour of " + name + " " + ROLE_NAMES[role], title: "Colour of the " + ROLE_NAMES[role].toLowerCase() + " polygons" });
      swatch.addEventListener("change", () => setNameColour([{ layer, count }], name, swatch.value));
      box.appendChild(el("summary", { class: "row annotation-part-head", title: (role === "good" ? "Good tissue examples for training. " : "Reviewed artifact polygons. ") + (layer.saved_file || "") }, [show, swatch, el("span", { class: "grow", text: ROLE_NAMES[role] }),
        el("span", { class: "kind", text: String(count), title: count + " polygons" }),
        el("label", { class: "inline-check", title: "Put these polygons in the exported GeoJSON" }, [tick, el("span", { text: "Export" })])]));
    } else if (parts.length > 1) box.appendChild(el("div", { class: "hint", text: annotationOrigin(layer) + " · " + count + " polygons" }));
    let loaded = false;
    const load = () => {
      if (!open || (byRole && !box.open) || loaded) return;
      loaded = true;
      const body = el("div", { class: "layer-body", "data-layer-controls": "right" });
      body.dataset.onlyName = name;
      regionsControls(layer, body);
      box.appendChild(body);
    };
    if (byRole) box.addEventListener('toggle', () => {
      if (!box.isConnected) return;
      const changed=roleState[roleKey]!==box.open;roleState[roleKey] = box.open; if(changed)saveView(); load();
    });
    load();
    holder.appendChild(box);
  });
  return holder;
}

function nameRow(name, parts, source) {
  const rows = state.annotationRowsOpen || (state.annotationRowsOpen = {});
  const rowKey = source + "::" + name;
  // Open when asked, or an artifact row while its stage draws into it (its review tools live there); rows start compact.
  const open = rows[rowKey] ?? parts.some(({ layer }) => layer.id === state.drawInto && (layer.artifact_part || layer.training_part));
  const card = el("div", { class: "layer annotation-name" + (open ? "" : " collapsed"), "data-annotation-name": name });
  const total = parts.reduce((sum, p) => sum + p.count, 0);
  const show = el("input", { type: "checkbox", "aria-label": "Show " + name, title: "Show or hide these polygons", 'data-annotation-visibility':'' });
  show.refreshVisibility=()=>{
  show.checked = parts.some(({ layer }) => nameShown(layer, name));
  show.indeterminate = show.checked && !parts.every(({ layer }) => nameShown(layer, name));
  };show.refreshVisibility();
  show.addEventListener("click", (event) => event.stopPropagation());
  show.addEventListener("change", () => parts.forEach(({ layer }) => setNameShown(layer, name, show.checked)));
  const byRole = parts.some((p) => annotationRole(p.layer));
  // Artifacts: the name's colour is the artifacts' (red); Good tissue has its own colour below.
  const coloured = byRole ? parts.filter((p) => annotationRole(p.layer) === "artifact") : parts;
  const first = coloured.flatMap(({ layer }) => (layer.features || []).filter((f) => annotationName(layer, f) === name))[0];
  const colour = el("input", { type: "color", value: first ? shapeColor(first) : nameColour((coloured[0] || parts[0]).layer, name), "aria-label": "Colour of " + name,
    title: "Colour of " + name + (byRole ? " artifacts (Good tissue has its own colour below)" : "") + " (kept in the file and in the export)" });
  colour.addEventListener("click", (event) => event.stopPropagation());
  colour.addEventListener("change", () => setNameColour(coloured.length ? coloured : parts, name, colour.value));
  const keys = parts.map(({ layer }) => annotationKey(layer, name));
  const tick = byRole ? null : el("input", { type: "checkbox", "aria-label": "Export " + name });
  if (tick) {
    tick.checked = keys.every((key) => exportChosen(key, ""));
    tick.addEventListener("click", (event) => event.stopPropagation());
    tick.addEventListener("change", () => keys.forEach((key) => setExportChosen(key, tick.checked)));
  }
  const toggle = () => { rows[rowKey] = !open; saveView(); renderShapes(); buildPanel(); };
  const fold = el("button", { class: "layer-fold annotation-fold", text: open ? "▾" : "▸", type: "button",
    "aria-label": "Details for " + name, "aria-expanded": String(!!open),
    onclick: event => { event.stopPropagation(); toggle(); } });
  const copy = el("button", { class: "btn tiny", text: "⧉", title: "Duplicate: a copy of these polygons under a new name, to rename further",
    "aria-label": "Duplicate " + name, onclick: (event) => { event.stopPropagation(); duplicateName(parts, name); } });
  const remove = el("button", { class: "btn tiny", text: "×", title: "Delete " + name + " (its polygons; a copied or imported layer leaves the project)",
    "aria-label": "Delete " + name, onclick: (event) => { event.stopPropagation(); deleteName(parts, name); } });
  const head = el("div", { class: "layer-head", onclick: toggle }, [fold, show, colour,
    el("span", { class: "name", text: name, title: name }), el("span", { class: "kind", text: total.toLocaleString()+(total === 1 ? " polygon" : " polygons"), title: "Individual polygons; expanding this list does not change their boundaries or rendering." }),
    tick ? el("label", { class: "inline-check", title: "Put these polygons in the exported GeoJSON, named " + name,
      onclick: (event) => event.stopPropagation() }, [tick, el("span", { text: "Export" })]) : null, copy, remove].filter(Boolean));
  const origins = Array.from(new Set(parts.map(({ layer }) => annotationOrigin(layer))));
  head.title = origins.join(' · ');
  if (open && state.studio && parts.length === 1 && parts[0].layer.annotation_group === 'imported') {
    head.querySelector('.name').after(el('button', {class:'btn small annotation-tissue-action', text:'Use as Tissue…',
      title:'Choose an annotation of this file to copy into the Tissue panel.',
      onclick:event => { event.stopPropagation(); tissueImportDialog(parts[0].layer); }}));
  }
  if (open) {
    const tip = help(origins.join(' · '));
    tip.addEventListener('click', event => event.stopPropagation());
    head.appendChild(tip);
  }
  card.append(head, partRows(name, parts, open, source));
  if (open && parts.length === 1 && !byRole) {
    const order = card.querySelector('[aria-label="Sort polygons"]');
    if (order) {
      order.classList.add('component-order');
      order.addEventListener('click', event => event.stopPropagation());
      head.querySelector('.name').after(order);
    }
  }
  return card;
}

// File identity is separate from the class/export name. Two imported copies can
// have the same name without sharing visibility, counts, or removal controls.
function annotationFileKey(layer) { return layer.saved_file || layer.token || layer.id; }

function annotationSourceHelp(layers, kind) {
  const paths = Array.from(new Set(layers.map(layer => layer.saved_file || layer.name)));
  const explanation = kind === 'Copy' ? 'Editable imported copy. Changes stay in this copy.' :
    kind === 'Linked' ? 'Linked GeoJSON. Edit a copy to change its polygons. Expand an annotation name to select individual polygons.' :
    'Working annotations drawn or generated for this image.';
  const tip = help(explanation + '\n' + paths.join('\n') + '\nVisibility only changes the view. Choose the segmentation source in Area.');
  tip.addEventListener('click', event => { event.preventDefault(); event.stopPropagation(); });
  return tip;
}

// Large class lists are filtered locally; no file reads or geometry changes.
function annotationClassFilter(branch, entries, key, changed = () => { saveView(); renderShapes(); buildPanel(); }) {
  if (entries.length <= 8) return;
  const searches = state.annotationClassSearch || (state.annotationClassSearch = {});
  const search = el('input', {type:'search', placeholder:'Find a class…', 'aria-label':'Find an annotation class', value:searches[key] || ''});
  const all = el('input', {type:'checkbox', 'aria-label':'Show matching classes'});
  const matches = () => entries.filter(entry => entry.name.toLocaleLowerCase().includes(search.value.trim().toLocaleLowerCase()));
  const sync = () => {
    const found = matches(), flags = found.map(entry => entry.shown());
    entries.forEach(entry => { entry.node.hidden = !found.includes(entry); });
    all.checked = flags.length > 0 && flags.every(Boolean);
    all.indeterminate = flags.some(Boolean) && !all.checked;
    all.disabled = !flags.length;
  };
  search.addEventListener('input', () => { searches[key] = search.value; sync(); });
  all.addEventListener('change', () => { const on = all.checked; matches().forEach(entry => entry.show(on, false)); changed(); });
  branch.appendChild(el('div', {class:'annotation-class-search'}, [all, search]));
  sync();
}

function annotationLinkedControls(layer, body, key) {
  body.appendChild(el('div', {class:'row annotation-file-actions'}, [
    el('button', {class:'btn small', text:'Edit a copy', onclick:() => importAnnotationCopy({source_token:layer.token, name:layer.name})}),
    el('button', {class:'btn small', text:'Use as Tissue…', onclick:() => tissueImportDialog(layer)}),
    ...[[1,'↑'],[-1,'↓']].map(([step,text]) => el('button', {class:'btn tiny', text,
      title:step === 1 ? 'Draw above the next layer' : 'Draw below the previous layer', onclick:() => moveLayer(layer, step)}))]));
  const list = el('div', {class:'annotation-classes', text:'Loading…'});
  body.appendChild(list);
  if (state.studio || state.reviewer && layer.review_item) { annotationComponents(layer, list); return; }
  const load = () => {
    list.textContent = 'Loading…';
    annotationLegend(layer).then(legend => {
      if (!body.isConnected) return;
      list.textContent = '';
      const entries = (legend.items || []).map(item => {
        const shown = () => layer.visible && !layer.hidden.has(item.value);
        const show = (on, redraw = true) => {
          if (on) {
            if (!layer.visible) layer.hidden = new Set(legend.items.filter(other => other.value !== item.value).map(other => other.value));
            else layer.hidden.delete(item.value);
            setVisible(layer, true);
          } else layer.hidden.add(item.value);
          if (redraw) annotationDisplayChanged(layer);
        };
        const check = el('input', {type:'checkbox', 'aria-label':'Show ' + item.value,'data-annotation-visibility':''});
        check.refreshVisibility=()=>{check.checked=shown();};check.refreshVisibility();check.addEventListener('change', () => show(check.checked));
        const node = el('label', {class:'annotation-linked-class'}, [check,
          el('span', {class:'swatch', style:'background:' + item.color}), el('span', {class:'name', text:item.value}),
          el('span', {class:'kind', text:Number(item.count).toLocaleString(), title:'Features (may contain multiple polygons)'})]);
        return {name:item.value, node, shown, show};
      });
      annotationClassFilter(list, entries, key, () => bump(layer));
      entries.forEach(entry => list.appendChild(entry.node));
      if (!entries.length) list.textContent = 'No classes.';
    }).catch(error => {
      if (body.isConnected) annotationLegendError(list, error, load);
    });
  };
  load();
}

// The Annotations group of Layers in Studio.
function annotationBlocks(group, shown) {
  const reviewLayers=shown.filter(layer=>layer.review_inline);
  shown=shown.filter(layer=>!layer.review_inline);
  const image = state.layers.find((l) => l.type === "image");
  const imageName = String(image ? image.name : "This image").replace(/(\.ome)?\.(tiff?|qptiff|svs)$/i, "");
  const sources = new Map([["", new Map()]]);
  shown.filter((layer) => layer.type === "shapes").forEach((layer) => {
    const source = annotationSource(layer) ? annotationFileKey(layer) : '';
    if (!sources.has(source)) sources.set(source, new Map());
    const names = sources.get(source);
    const counts = new Map();
    (layer.features || []).forEach((f) => { const name = annotationName(layer, f); counts.set(name, (counts.get(name) || 0) + 1); });
    // Tissue and artifacts keep their row while empty, so there is somewhere to draw and review.
    if (!counts.size && (layer.artifact_part || layer.training_part || layer.annotation_kind === "tissue" || layer.working_tissue)) {
      counts.set(annotationName(layer, { properties: {} }), 0);
    }
    counts.forEach((count, name) => { if (!names.has(name)) names.set(name, []); names.get(name).push({ layer, count }); });
  });
  const blocks = state.annotationBlocksOpen || (state.annotationBlocksOpen = {});
  let working;
  const sections = new Map(), categories = new Map();
  const section = name => {
    if (!sections.has(name)) {
      const branch = el('details', {class:'layer-group '+(name==='Review polygons'?'annotation-subgroup':'annotation-section'), 'data-annotation-section':name});
      if(name==='Review polygons')branch.dataset.annotationRoot='review';
      branch.open = blocks['section::'+name] ?? true;
      branch.append(el('summary', {}, [el('span', {class:'name', text:name})]));
      branch.addEventListener('toggle', () => {if(branch.isConnected&&blocks['section::'+name]!==branch.open){blocks['section::'+name]=branch.open;saveView();}});
      working.append(branch);sections.set(name,branch);
    }
    return sections.get(name);
  };
  let imports;
  const imported = () => {
    if (!imports) {
      imports=el('details',{class:'layer-group annotation-subgroup', 'data-annotation-root':'imports'});
      imports.open=blocks['imports-root'] ?? true;
      imports.append(el('summary',{},[el('span',{class:'name',text:'Imported GeoJSON'})]));
      imports.addEventListener('toggle',()=>{if(imports.isConnected&&blocks['imports-root']!==imports.open){blocks['imports-root']=imports.open;saveView();}});
      group.append(imports);
    }
    return imports;
  };
  sources.forEach((names, source) => {
    if (!names.size && source) return;
    const layers = Array.from(new Set(Array.from(names.values()).flat().map((p) => p.layer)));
    if(!source) layers.push(...shown.filter(l=>l.type==='annotations' && l.annotation_origin==='local' && l.annotation_section));
    const sourceName = source ? annotationSource(layers[0]) : imageName;
    const peers = source ? state.layers.filter(layer => layer.type === 'shapes' && annotationSource(layer) === sourceName) : [];
    const identity = source ? (String(layers[0].saved_file || '').split('/').pop().match(/^([a-f0-9]{8})_/)?.[1] || layers[0].id) : '';
    const title = (source ? sourceName : 'Working annotations') + (peers.length > 1 ? ' · ' + identity : '');
    const branch = el("details", { class: "layer-group annotation-subgroup", "data-annotation-source": title });
    branch.dataset.annotationKey = source || 'working';
    branch.open = blocks[source] ?? blocks[sourceName] ?? !source;
    const polygons = Array.from(names.values()).flat().reduce((sum, p) => sum + p.count, 0);
    const drop = source ? el("button", { class: "btn tiny", text: "×", title: "Remove this imported file from the project (kept in outputs/studio/deleted)",
      "aria-label": "Remove " + title, onclick: async (event) => {
        event.preventDefault(); event.stopPropagation();
        if (!window.confirm("Remove the imported " + title + " (" + polygons + " polygons) from this image? Its copy is kept in outputs/studio/deleted.")) return;
        for (const layer of layers) {
          try { await studioPost("/annotation_sources", { action: "remove_layer", image_token: image.token, layer_token: layer.token }); removeLayer(layer, true); }
          catch (error) { message(error.message, true); return; }
        }
        buildPanel(); message(title + " removed.");
      } }) : null;
    const kind = source ? 'Copy' : 'Working';
    branch.appendChild(el("summary", {}, [annotationVisibility(layers, title), el("span", { class: "name", text: title, title:source ? layers.map(l=>l.saved_file || l.name).join('\n') : imageName }),
      ...(source ? [el('span', {class:'annotation-source-kind', text:kind})] : []),
      el("span", { class: "group-count", text: polygons.toLocaleString(), title: names.size + " classes · " + polygons + " polygons" }),
      annotationSourceHelp(layers, kind), drop].filter(Boolean)));
    branch.addEventListener("toggle", () => { if (branch.isConnected&&blocks[source]!==branch.open){blocks[source] = branch.open;saveView();} });
    const entries = Array.from(names.entries()).sort((a, b) => nameOrder(a[0], a[1]).localeCompare(nameOrder(b[0], b[1]), undefined, { numeric: true }))
      .map(([name, parts]) => ({name, node:nameRow(name, parts, source), shown:() => parts.every(({layer}) => nameShown(layer, name)),
        show:(on, redraw) => parts.forEach(({layer}) => setNameShown(layer, name, on, redraw))}));
    if (source) {
      annotationClassFilter(branch, entries, source);
      entries.forEach(entry => branch.appendChild(entry.node));
      imported().appendChild(branch);
    } else {
      working=branch;branch.dataset.annotationRoot='working';group.append(branch);
      if(reviewLayers.length){
        const review=section('Review polygons'),summary=review.querySelector('summary');
        summary.prepend(annotationVisibility(reviewLayers,'Review polygons'));
        summary.append(help('All flagged annotation polygons in this image. Check rows to select them; the header controls visibility. Visible independently of their source families. Proposed labels use the destination family colour; source annotations change only after owner approval.'));
        const rows=el('div',{class:'component-page review-polygon-list'}),pager=el('div',{class:'row'}),features=reviewLayers.flatMap(layer=>(layer.features||[]).map(feature=>({layer,feature})));
        const order=select([['place','By place'],['size_down','Largest first'],['size_up','Smallest first']],state.reviewPolygonSort||'size_down',value=>{
          state.reviewPolygonSort=value;state.reviewPolygonOffset=offset=0;draw();saveView();
        });
        order.classList.add('component-order');order.setAttribute('aria-label','Sort review polygons');order.onclick=e=>e.stopPropagation();
        summary.insertBefore(order,summary.lastChild);
        let offset=Math.min(state.reviewPolygonOffset||0,Math.max(0,Math.floor((features.length-1)/100)*100));
        const draw=()=>{
          rows.replaceChildren();
          features.sort((a,b)=>{
            const x=a.feature.properties.review_area,y=b.feature.properties.review_area;
            if(order.value!=='place')return (x==null)-(y==null)||(order.value==='size_down'?-1:1)*((x||0)-(y||0))||String(a.feature.properties.id).localeCompare(String(b.feature.properties.id));
            const ab=featureBounds(a.feature),bb=featureBounds(b.feature);return ab.y0-bb.y0||ab.x0-bb.x0||String(a.feature.properties.id).localeCompare(String(b.feature.properties.id));
          });
          features.slice(offset,offset+100).forEach(({layer,feature})=>{
            const name=annotationName(layer,feature),colour=shapeColor(feature),show=el('input',{type:'checkbox','aria-label':'Select review polygon '+name});
            show.checked=annotationSelection(layer).has(feature.properties.id);
            show.onchange=()=>{selectAnnotations(layer,[feature.properties.id],{toggle:true});};
            const focus=el('button',{class:'review-polygon-name',text:name,title:'Show polygon · '+REVIEW_STATES[feature.properties.review_state],onclick:event=>{
              layer.visible=true;layer.hiddenReviewIds?.delete(feature.properties.id);
              const additive=event.ctrlKey||event.metaKey||event.shiftKey;
              selectAnnotations(layer,[feature.properties.id],{replace:!additive,toggle:additive});zoomTo(feature);draw();
            }});
            rows.append(el('div',{class:'row','data-review-polygon':feature.properties.review_item,'data-review-feature':feature.properties.id},[show,el('span',{class:'component-colour',style:'background:'+colour}),focus,...(feature.properties.review_area==null?[]:[el('span',{class:'muted area',text:areaText(feature.properties.review_area)})]),reviewPolygonChoices(feature)]));
          });
          pager.replaceChildren(el('span',{class:'muted',text:features.length+' polygons'}));
          if(!state.reviewer)pager.append(el('button',{class:'btn tiny',text:'Remove from review',
            title:'Remove selected review entries; keep their source polygons and saved edits.',
            onclick:()=>removeReviewItems(features.filter(({layer,feature})=>annotationSelection(layer).has(feature.properties.id)).map(({feature})=>feature.properties.review_item))}));
          if(features.length>100)pager.append(el('button',{class:'btn tiny',text:'←',disabled:offset?null:'',onclick:()=>{state.reviewPolygonOffset=offset=Math.max(0,offset-100);draw();}}),el('button',{class:'btn tiny',text:'→',disabled:offset+100<features.length?null:'',onclick:()=>{state.reviewPolygonOffset=offset+=100;draw();}}));
        };
        review.append(rows,pager);draw();
      }
      entries.forEach(entry => {
        const parts=names.get(entry.name), layer=parts[0].layer;
        const kind=parts.some(p=>p.layer.artifact_part || p.layer.training_part) ? 'Artifacts' :
          parts.some(p=>p.layer.annotation_kind==='tissue' || p.layer.working_tissue) ? 'Tissue' : '';
        (kind ? section(kind) : working).appendChild(entry.node);
        if(kind==='Artifacts')categories.set(entry.name,entry.node.querySelector('.annotation-name-parts'));
      });
    }
  });
  // Files read as they are (classic outputs, zones): each its own block (once), with its classes.
  const listed = new Set(shown.filter(l=>l.type==='shapes' && l.saved_file).map(annotationFileKey));
  shown.filter((layer) => layer.type === "annotations").forEach((layer) => {
    const key = annotationFileKey(layer);
    if (listed.has(key)) return;
    listed.add(key);
    const branch = el("details", { class: "layer-group annotation-subgroup", "data-annotation-source": layer.name });
    branch.dataset.annotationKey = key;
    branch.dataset.annotationToken = layer.token;
    branch.open = blocks[key] ?? !!blocks[layer.name];
    const hide = el("button", { class: "btn tiny", text: "×", title: "Take this file off this image's annotations (the file stays on disk)",
      "aria-label": "Remove " + layer.name, onclick: (event) => { event.preventDefault(); event.stopPropagation(); hideFile(layer); } });
    const basename = value => String(value).split('/').pop().trim();
    const sameNames = state.layers.filter(other => other.type === 'annotations' && basename(other.name) === basename(layer.name));
    const location = String(layer.saved_file || layer.name).split('/').slice(0,-1).join('/');
    const local=layer.annotation_origin==='local' && layer.annotation_section;
    const title = (local && layer.annotation_role==='artifact' ? 'Bad tissue' : basename(layer.name)) +
      (sameNames.length > 1 ? ' · ' + (basename(location) || layer.id) : '');
    branch.appendChild(el("summary", {}, [annotationVisibility([layer], layer.name), el("span", { class: "name", text: title, title:layer.saved_file || layer.name }),
      el('span', {class:'annotation-source-kind', text:local ? 'Saved' : layer.annotation_origin==='copy' ? 'Copy' : 'Linked'}),
      annotationSourceHelp([layer], local ? 'Working' : 'Linked'), hide]));
    let loaded = false;
    const open = () => {
      if (!branch.isConnected) return;
      if(blocks[key]!==branch.open){blocks[key] = branch.open;saveView();}
      if (branch.open && !loaded) {
        loaded = true;
        const body = el('div', {class:'annotation-file-body', 'data-layer-id':layer.id});
        branch.appendChild(body); annotationLinkedControls(layer, body, key);
      }
    };
    branch.addEventListener("toggle", open);
    if (local) {
      let target=section(layer.annotation_section);
      if(layer.annotation_category){
        if(!categories.has(layer.annotation_category)){
          const category=el('details',{class:'layer-group annotation-section', 'data-annotation-name':layer.annotation_category});
          category.open=blocks['category::'+layer.annotation_category] ?? false;
          category.append(el('summary',{},[el('span',{class:'name',text:layer.annotation_category})]));
          category.addEventListener('toggle',()=>{if(category.isConnected&&blocks['category::'+layer.annotation_category]!==category.open){blocks['category::'+layer.annotation_category]=category.open;saveView();}});
          target.append(category);categories.set(layer.annotation_category,category);
        }
        target=categories.get(layer.annotation_category);
      }
      target.append(branch);
    } else imported().append(branch);
    // The group is attached by buildPanel after this function returns.
    queueMicrotask(open);
  });
  if(sections.has('Review polygons'))group.append(sections.get('Review polygons'));
}
