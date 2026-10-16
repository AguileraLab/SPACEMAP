// Studio's Annotations in Layers: one block for the open image and one for each imported file, then one row per
// annotation name (tissue, Blood clots, Epithelia, Tumor…), whatever layer file its polygons are kept in. A row shows or
// hides its polygons, sets their colour, says where they came from, ticks them for Export, and lists them (select, Go,
// label, delete) with the layer's own tools. Artifacts split into Artifact and Good tissue (training examples).

const ROLE_NAMES = { artifact: "Artifact", good: "Good tissue (training)" };

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
function setNameShown(layer, name, on) {
  const all = namesOf(layer);
  if (on) {
    if (!layer.visible) { all.delete(name); layer.hiddenNames = all; setVisible(layer, true); }
    else if (layer.hiddenNames) layer.hiddenNames.delete(name);
  } else {
    layer.hiddenNames = layer.hiddenNames || new Set();
    layer.hiddenNames.add(name);
    if (!all.size || Array.from(all).every((n) => layer.hiddenNames.has(n))) { layer.hiddenNames = null; setVisible(layer, false); }
  }
  saveView(); renderShapes(); buildPanel();
}

function annotationVisibility(layers, title) {
  const show = el('input', {type:'checkbox', 'aria-label':'Show all annotations in ' + title,
    title:'Show or hide all annotations'});
  const flags = layers.flatMap(layer => layer.type === 'shapes' ?
    Array.from(namesOf(layer), name => nameShown(layer, name)) : [layer.visible]);
  show.checked = flags.length > 0 && flags.every(Boolean) && !layers.some(layer => layer.type === 'annotations' && layer.hidden?.size);
  show.indeterminate = flags.some(Boolean) && !show.checked;
  // Read-only GeoJSON classes are loaded lazily, unlike editable polygons.
  if (layers.length === 1 && layers[0].type === 'annotations') {
    const layer = layers[0];
    annotationLegend(layer).then(legend => {
      if (!show.isConnected) return;
      const shown = (legend.items || []).map(item => layer.visible && !layer.hidden.has(item.value));
      show.checked = shown.length > 0 && shown.every(Boolean);
      show.indeterminate = shown.some(Boolean) && !show.checked;
    }).catch(() => {}); // the layer's legend reports its own read errors
  }
  show.disabled = !layers.length;
  show.addEventListener('click', event => event.stopPropagation());
  show.addEventListener('change', () => {
    const on = show.checked;
    layers.forEach(layer => {
      layer.hiddenNames = null;
      if (on && layer.type === 'annotations') layer.hidden = new Set();
      setVisible(layer, on);
      if (on && layer.type === 'annotations') refreshLayer(layer);
    });
    saveView(); renderShapes(); buildPanel();
  });
  return show;
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

function partRows(name, parts, open, source) {
  const holder = el("div", { class: "annotation-name-parts" });
  const byRole = parts.some((p) => annotationRole(p.layer));
  parts.forEach(({ layer, count }) => {
    const role = annotationRole(layer);
    const box = el("div", { class: "annotation-name-part", "data-layer-id": layer.id });
    if (byRole) {
      const key = annotationKey(layer, name);
      const show = el("input", { type: "checkbox", "aria-label": "Show " + name + " " + ROLE_NAMES[role] });
      show.checked = nameShown(layer, name);
      show.addEventListener("change", () => setNameShown(layer, name, show.checked));
      const tick = el("input", { type: "checkbox", "aria-label": "Export " + name + " " + ROLE_NAMES[role] });
      tick.checked = exportChosen(key, role);
      tick.addEventListener("change", () => setExportChosen(key, tick.checked));
      const own = (layer.features || []).find((f) => annotationName(layer, f) === name);
      const swatch = el("input", { type: "color", value: own ? shapeColor(own) : role === "good" ? "#2ecc71" : "#e6194b",
        "aria-label": "Colour of " + name + " " + ROLE_NAMES[role], title: "Colour of the " + ROLE_NAMES[role].toLowerCase() + " polygons" });
      swatch.addEventListener("change", () => setNameColour([{ layer, count }], name, swatch.value));
      box.appendChild(el("div", { class: "row annotation-part-head" }, [show, swatch, el("span", { class: "grow", text: ROLE_NAMES[role] }),
        el("span", { class: "kind", text: count + " polygons" }),
        el("label", { class: "inline-check", title: "Put these polygons in the exported GeoJSON" }, [tick, el("span", { text: "Export" })])]));
    } else if (parts.length > 1) box.appendChild(el("div", { class: "hint", text: annotationOrigin(layer) + " · " + count + " polygons" }));
    if (open && layer.annotation_group === "imported" && state.studio) box.appendChild(el("button", { class: "btn small", text: "Use as Tissue…",
      title: "Choose an annotation of this file to copy into the Tissue panel.", onclick: () => tissueImportDialog(layer) }));
    if (open) {
      const body = el("div", { class: "layer-body", "data-layer-controls": "right" });
      body.dataset.onlyName = name;
      regionsControls(layer, body);
      box.appendChild(body);
    }
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
  const show = el("input", { type: "checkbox", "aria-label": "Show " + name, title: "Show or hide these polygons" });
  show.checked = parts.some(({ layer }) => nameShown(layer, name));
  show.indeterminate = show.checked && !parts.every(({ layer }) => nameShown(layer, name));
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
  const fold = el("span", { class: "layer-fold", text: open ? "▾" : "▸", "aria-hidden": "true" });
  const copy = el("button", { class: "btn tiny", text: "⧉", title: "Duplicate: a copy of these polygons under a new name, to rename further",
    "aria-label": "Duplicate " + name, onclick: (event) => { event.stopPropagation(); duplicateName(parts, name); } });
  const remove = el("button", { class: "btn tiny", text: "×", title: "Delete " + name + " (its polygons; a copied or imported layer leaves the project)",
    "aria-label": "Delete " + name, onclick: (event) => { event.stopPropagation(); deleteName(parts, name); } });
  const head = el("div", { class: "layer-head", onclick: () => { rows[rowKey] = !open; buildPanel(); } }, [show, fold, colour,
    el("span", { class: "name", text: name, title: name }), el("span", { class: "kind", text: String(total), title: total + (total === 1 ? " polygon" : " polygons") }),
    tick ? el("label", { class: "inline-check", title: "Put these polygons in the exported GeoJSON, named " + name,
      onclick: (event) => event.stopPropagation() }, [tick, el("span", { text: "Export" })]) : null, copy, remove].filter(Boolean));
  const origins = Array.from(new Set(parts.map(({ layer }) => annotationOrigin(layer))));
  head.title = origins.join(' · ');
  card.append(head, ...(open ? [el("div", { class: "annotation-origin muted", text: origins.join(" · ") })] : []), partRows(name, parts, open, source));
  return card;
}

// The Annotations group of Layers in Studio.
function annotationBlocks(group, shown) {
  const image = state.layers.find((l) => l.type === "image");
  const imageName = String(image ? image.name : "This image").replace(/(\.ome)?\.(tiff?|qptiff|svs)$/i, "");
  const sources = new Map([["", new Map()]]);
  shown.filter((layer) => layer.type === "shapes").forEach((layer) => {
    const source = annotationSource(layer);
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
  sources.forEach((names, source) => {
    if (!names.size && source) return;
    const title = source || imageName;
    const branch = el("details", { class: "layer-group annotation-subgroup", "data-annotation-source": title });
    branch.open = blocks[title] ?? !source;
    const polygons = Array.from(names.values()).flat().reduce((sum, p) => sum + p.count, 0);
    const layers = Array.from(new Set(Array.from(names.values()).flat().map((p) => p.layer)));
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
    branch.appendChild(el("summary", {}, [annotationVisibility(layers, title), el("span", { class: "name", text: title, title: source ? "Imported file" : "The open image" }),
      el("span", { class: "group-count", text: String(polygons), title: names.size + " names · " + polygons + " polygons" }), drop].filter(Boolean)));
    branch.addEventListener("toggle", () => { if (branch.isConnected) blocks[title] = branch.open; });
    Array.from(names.entries()).sort((a, b) => nameOrder(a[0], a[1]).localeCompare(nameOrder(b[0], b[1]), undefined, { numeric: true }))
      .forEach(([name, parts]) => branch.appendChild(nameRow(name, parts, source)));
    if (!names.size) branch.appendChild(el("p", { class: "hint", text: "No annotations yet." }));
    group.appendChild(branch);
  });
  // Files read as they are (classic outputs, zones): each its own block (once), with its classes.
  const listed = new Set();
  shown.filter((layer) => layer.type === "annotations").forEach((layer) => {
    if (listed.has(layer.token) || listed.has(layer.name)) return;
    listed.add(layer.token); listed.add(layer.name);
    const branch = el("details", { class: "layer-group annotation-subgroup", "data-annotation-source": layer.name });
    branch.open = !!blocks[layer.name];
    const hide = el("button", { class: "btn tiny", text: "×", title: "Take this file off this image's annotations (the file stays on disk)",
      "aria-label": "Remove " + layer.name, onclick: (event) => { event.preventDefault(); event.stopPropagation(); hideFile(layer); } });
    branch.appendChild(el("summary", {}, [annotationVisibility([layer], layer.name), el("span", { class: "name", text: layer.name }), hide]));
    branch.addEventListener("toggle", () => { if (branch.isConnected) blocks[layer.name] = branch.open; });
    branch.appendChild(layerCard(layer));
    group.appendChild(branch);
  });
}
