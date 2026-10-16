/* SPACEMAP viewer, part 8 of 12: the side panel and its layer cards.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- panel
function buildPanel() {
  syncSplitMarkers();
  updateToolbar();
  const panel = $("panel");
  const restoreLabelFocus=keepRegionLabelFocus(panel);
  const scroll = panel.scrollTop;
  panel.innerHTML = "";
  const filters = el('div', {class:'layer-filter', role:'group', 'aria-label':'Layers to list'});
  [['All', false], ['Shown', true]].forEach(([label, shown]) => filters.appendChild(el('button', {
    class:'btn small', text:label, 'aria-pressed':String(!!state.layersShownOnly === shown),
    title:shown ? 'List only layers shown on the image. This does not change the image.' : 'List all layers, including hidden ones.',
    onclick:() => { state.layersShownOnly = shown; buildPanel(); saveView(); }
  })));
  panel.appendChild(el("div", { class: "row section-title-row" }, [el("div", { class: "section-title", text: "Layers" }),
    help("Tick to show a layer on the image. Open a group or click a layer name for its settings; folding never hides the image. " +
      "All includes hidden layers; Shown lists only visible ones. Arrows change which layer overlaps another. Saved files are in Outputs."), filters]));
  // Keep the image controls first. Grouping changes the list, never the drawing order.
  const ordered = state.layers.slice().reverse().filter((layer) => !mergedHE(layer));
  const listed = layer => !state.layersShownOnly || layer.visible;
  ordered.filter(layer => layer.type === 'image' && listed(layer)).forEach(layer => panel.appendChild(layerCard(layer)));
  const groups = state.layerGroups || (state.layerGroups = {});
  [['cells', 'Cells and maps', ['cells','density']], ['annotations', 'Annotations', ['shapes','annotations']]].forEach(([key, title, types]) => {
    const members = ordered.filter(layer => types.includes(layer.type));
    if (!members.length && !(key === 'cells' && state.studio)) return;  // Studio: Import cells is there
    const group = el('details', {class:'layer-group', 'data-layer-group':key});
    group.open = !!groups[key];
    group.addEventListener('toggle', () => { if (group.isConnected) { groups[key] = group.open; saveView(); } });
    group.appendChild(el('summary', {onclick:event => {
      event.preventDefault(); group.open = !group.open; groups[key] = group.open; saveView();
    }}, [el('span', {class:'name', text:title}),
      el('span', {class:'group-count', text:members.filter(l => l.visible).length + ' shown / ' + members.length})]));
    const shown = members.filter(listed);
    if (key === 'annotations') group.appendChild(annotationMarkerControls());
    if (key === 'cells' && members.some(layer => layer.type === 'cells')) group.appendChild(cellMarkerControls());
    if (key === 'annotations' && state.studio) {
      annotationBlocks(group, shown);  // by image or file, then by name (annotation-rows.js)
    } else if (key === 'cells' && state.studio) {
      cellBlocks(group, shown);  // outlines, colours and maps as rows (cell-rows.js)
    } else shown.forEach(layer => group.appendChild(layerCard(layer)));
    if (!shown.length && members.length) group.appendChild(el('p', {class:'hint', text:'These layers are hidden. Choose All above to show their controls.'}));
    if (key === 'annotations' && state.studio) group.appendChild(annotationImportControls());
    panel.appendChild(group);
  });
  panel.appendChild(addCard());
  const cells = state.layers.find((l) => l.type === "cells" && (l.tables || []).length);
  if (cells && !state.studio) {
    const anchor = el("div");
    panel.appendChild(anchor);
    const showLabels = () => {
      if (anchor.isConnected && state.layers.includes(cells)) anchor.replaceWith(labellingCard(cells));
    };
    if (state.imagesReady) showLabels();
    else state.whenImagesReady.then(showLabels);
  }
  panel.scrollTop = scroll;
  state.studio?.refreshLive();
  restoreLabelFocus();
}

// The registered H&E is three channels of the same image as the markers: it is
// controlled from that image's card rather than listed as a second layer.
function mergedHE(layer) {
  return layer.id === "he" && state.layers.some((other) => other !== layer && other.type === "image" && other.token === layer.token);
}

// Registration check: show the H&E only left of a vertical line (the swipe), 0-1 of the width.
function applySwipe(layer) {
  if (!layer.item || layer.swipe === undefined || layer.swipe === null) return;
  const size = layer.item.getContentSize();
  layer.item.setClip(layer.swipe >= 1 ? null : new OpenSeadragon.Rect(0, 0, size.x * layer.swipe, size.y));
}

function heCompare(options) {
  const he = state.layers.find((l) => l.id === "he" && l.type === "image");
  if (!he) return null;
  if (options) {
    if (options.opacity !== undefined) setOpacity(he, options.opacity);
    if (options.swipe !== undefined) { he.swipe = options.swipe; applySwipe(he); }
    if (options.visible !== undefined && options.visible !== he.visible) setVisible(he, options.visible);
    if (options.checker !== undefined && options.checker !== (he.checker || 0)) {
      he.checker = options.checker || 0;
      if (he.checker) { he.swipe = 1; applySwipe(he); }
      refreshLayer(he);
    }
    // Aligning by eye (a pair's raw H&E over its CODEX): turn, scale by the pixel sizes, move, start again.
    if (he.align) {
      if (options.rotation !== undefined) he.align.rotation = Number(options.rotation) || 0;
      if (options.scaled !== undefined) he.align.scaled = !!options.scaled;
      if (options.reset) Object.assign(he.align, { x: 0, y: 0, rotation: 0, scaled: true, zoom: 1 });
      if (options.place) Object.assign(he.align, { scaled: true, ...options.place });  // Quick align: {x, y, rotation, zoom}
      applyAlign(he);
    }
    if (options.move !== undefined) state.alignMove = !!options.move && !!he.align;
    if (options.codexRotation !== undefined && state.viewer) state.viewer.viewport.setRotation(Number(options.codexRotation) || 0);
  }
  return { visible: !!he.visible, opacity: he.opacity, swipe: he.swipe === undefined ? 1 : he.swipe, checker: he.checker || 0,
    align: he.align ? { ...he.align } : null, move: !!state.alignMove,
    codexRotation: state.viewer ? state.viewer.viewport.getRotation() : 0 };
}

function heControls(he) {
  const on = el("input", { type: "checkbox", "aria-label": "Show the registered H&E" });
  on.checked = !!he.visible;
  on.addEventListener("change", () => setVisible(he, on.checked));
  const opacity = el("input", { type: "range", min: "0", max: "100", value: String(Math.round(he.opacity * 100)),
    "aria-label": "H&E opacity" });
  opacity.addEventListener("input", () => { setOpacity(he, opacity.value / 100); if (!he.visible) { on.checked = true; setVisible(he, true); } });
  return field("H&E", el("div", { class: "inline" }, [on, opacity]),
    "The registered H&E stored in this image, in true colour over the markers. Tick to show it; the slider sets how much it covers them.");
}

function layerCard(layer) {
  const card = el("div", { class: "layer" + (layer.open ? "" : " collapsed"), 'data-layer-id':layer.id });
  const check = el("input", { type: "checkbox", title: "Show or hide this layer", 'aria-label':'Show ' + layer.name });
  check.checked = layer.visible;
  check.addEventListener("click", (e) => e.stopPropagation());
  check.addEventListener("change", () => { setVisible(layer, check.checked); buildPanel(); });
  const kinds = { image: layer.info.rgb ? "RGB image" : "Image", cells: "Cells", annotations: "Annotations", shapes: "Shapes", density: "Density" };
  const fold = el('span', {class:'layer-fold', text:layer.open ? '▾' : '▸', 'aria-hidden':'true'});
  const head = el("div", { class: "layer-head", onclick: () => {
    layer.open = !layer.open; card.classList.toggle("collapsed");
    fold.textContent = layer.open ? '▾' : '▸';
    if (layer.open) loadAnnotationLegend();
    saveView();
  } },
    [check, fold, el("span", { class: "name", text: layer.name, title: layer.name }),
      el("span", { class: "kind", text: layer.type === 'shapes' ?
        (layer.preview ? 'Preview · ' : '') + (layer.features || []).length + ' polygons' : kinds[layer.type] })]);
  if (layer.type !== "shapes") {
    const move = (step, label, title) => el("button", { class: "btn tiny", text: label, title: title,
      onclick: (e) => { e.stopPropagation(); moveLayer(layer, step); } });
    head.append(move(1, "\u2191", "Move up: draw above the layer over it"), move(-1, "\u2193", "Move down: draw below the layer under it"));
  }
  if (layer.added) {
    head.appendChild(el("button", { class: "btn tiny", text: "\u00d7", title: "Remove this layer from the view",
      onclick: (e) => { e.stopPropagation(); removeLayer(layer); } }));
  }
  const body = el("div", { class: "layer-body", "data-layer-controls":"right" });
  const opacity = el("input", { type: "range", min: "0", max: "100", value: String(Math.round(layer.opacity * 100)) });
  opacity.addEventListener("input", () => setOpacity(layer, opacity.value / 100));
  body.appendChild(el("div", { class: "row" }, [el("label", { class: "lbl", text: "Opacity" }), opacity]));
  const he = layer.type === "image" && layer.id !== "he" && state.layers.find((l) => l.id === "he" && l !== layer && l.token === layer.token);
  if (he) body.appendChild(heControls(he));
  if (layer.type === "image" && !layer.info.rgb) body.appendChild(channelList(layer));
  let legendLoaded = false;
  function loadAnnotationLegend() {
    if (layer.type !== "annotations" || legendLoaded) return;
    legendLoaded = true;
    const showLegend = () => legendInto(layer, body, "/annotations/" + layer.token + "/legend");
    if (state.imagesReady) showLegend();
    else state.whenImagesReady.then(() => {
      if (body.isConnected && state.layers.includes(layer)) showLegend();
    });
  }
  if (layer.open) loadAnnotationLegend();
  if (layer.type === "annotations" && state.studio) body.appendChild(el('button', {class:'btn small',text:'Edit a copy',
    title:'Keep this file intact and import its polygons into an editable Studio layer.',onclick:()=>importAnnotationCopy({source_token:layer.token,name:layer.name})}));
  if (state.studio && (layer.type === 'annotations' || layer.annotation_group === 'imported')) body.appendChild(el('button', {
    class:'btn small',text:'Use as Tissue…',title:'Choose an annotation class to copy into the Tissue panel.',onclick:()=>tissueImportDialog(layer)}));
  if (layer.type === "shapes") (layer.regions ? regionsControls : shapesControls)(layer, body);
  const showControls = () => {
    if (layer.type === "cells") {
      const controls = el('div'); body.appendChild(controls);
      layer.refreshCellControls = () => { controls.innerHTML = ''; cellControls(layer, controls); };
      layer.refreshCellControls();
    }
    else if (layer.type === "density") densityControls(layer, body);
  };
  if (layer.type === "cells" || layer.type === "density") {
    if (state.imagesReady) showControls();
    else state.whenImagesReady.then(() => {
      if (body.isConnected && state.layers.includes(layer)) showControls();
    });
  }
  card.appendChild(head);
  card.appendChild(body);
  return card;
}

function channelList(layer) {
  const wrap = el("div");
  const filter = el("input", { type: "text", placeholder: "Find a marker..." });
  const list = el("div", { class: "channels" });
  const shownNow = () => {
    const query = filter.value.trim().toLowerCase();
    return layer.channels.filter((c) => !query || c.name.toLowerCase().includes(query) || (c.label || "").toLowerCase().includes(query));
  };
  const all = allBox("All markers", async (on) => {
    const targets = shownNow();
    targets.forEach((c) => { c.visible = on; });
    if (on) await fillLimits(layer, targets.filter((c) => c.low === undefined || c.high === undefined));
    remember(layer);
    refreshLayer(layer);
    draw();
  });
  layer.syncAllMarkers = () => all.sync(shownNow().map((c) => c.visible));
  const draw = () => {
    list.innerHTML = "";
    shownNow().forEach((channel) => list.appendChild(channelRow(layer, channel)));
    layer.syncAllMarkers();
  };
  filter.addEventListener("input", draw);
  if (layer.display_token) wrap.appendChild(markerSets(layer));
  wrap.appendChild(autoControls(layer, {displayRange:true}));
  filter.setAttribute("aria-label", "Find a marker");
  wrap.appendChild(field("Markers", filter, "Tick markers to show them. Min is the background level (values at or " +
    "below it are black), max the brightness. Auto sets one marker's min from the image. Click a marker's name " +
    "for its histogram and to rename it."));
  // The All box, then the extra columns and the file's colours beside it (outside its label, so a click there is not a tick).
  const top = el("div", { class: "row all-top" }, [all.row]);
  wrap.appendChild(top);
  // Dye and exposure columns, when the image file records them.
  if ((layer.info.meta || []).some((m) => m && (m.dye || m.exposure_ms !== null && m.exposure_ms !== undefined))) {
    [["dye", "Dye"], ["exposure", "Exposure"]].forEach(([key, text]) => {
      const box = el("input", { type: "checkbox", "aria-label": "Show the " + text.toLowerCase() + " column" });
      box.checked = !!markerColumns()[key];
      box.addEventListener("change", () => { setMarkerColumn(key, box.checked); draw(); });
      top.appendChild(el("label", { class: "col-toggle", title: "Show each marker's " + text.toLowerCase() + " (from the image file)" }, [box, el("span", { text: text })]));
    });
  }
  // The colours the image file records (as QuPath shows them), in place of colours chosen or saved here.
  const fileColours = (layer.info.colors || []).filter(Boolean).length;
  if (fileColours) {
    top.appendChild(el("button", { class: "btn tiny", text: "Colours from the image", title: "Use each marker's colour as the image file records it (as QuPath shows it)",
      onclick: () => {
        layer.channels.forEach((c) => { const colour = layer.info.colors[c.index]; if (colour) { c.color = colour; c.custom = false; } });
        remember(layer); refreshLayer(layer); draw(); buildPanel();
        message("Each marker has its colour from the image file.");
      } }));
  }
  wrap.appendChild(list);
  layer.histogramHolder = el("div");
  wrap.appendChild(layer.histogramHolder);
  draw();
  showHistogram(layer);
  return wrap;
}

function liveForImage(image) {
  return state.layers.find((l) => l.type === "cells" && l.image_token === image.token && l.live);
}

function linkedMarker(image, channel) {
  const cells = liveForImage(image);
  return cells && cells.live.markers.find((m) => m.marker === channel.name);
}

function setMarkerRange(image, channel, low, high) {
  const deferredHigh = high === undefined && !channel.visible;
  if (!Number.isFinite(low) || (!deferredHigh && (!Number.isFinite(high) || high <= low))) {
    message("Enter valid numbers. Max must be greater than min.", true);
    return false;
  }
  delete image.thresholdPreview;
  channel.low = low; channel.high = high;
  const parts = (image.inputs || {})[channel.index];
  if (parts) { parts.low.value = low; parts.high.value = high === undefined ? "" : high; }
  const marker = linkedMarker(image, channel);
  if (marker && channel.resolveLinked !== false && marker.threshold !== low) {
    marker.threshold = low; marker.source = "changed";
    sendLive(liveForImage(image), { [marker.marker]: low });
  }
  remember(image); refreshLayer(image);
  if (image.redrawHistogram) image.redrawHistogram();
  if (image.syncLevels) image.syncLevels();
  return true;
}

function markerHigh(channel, low) {
  // Hidden markers have not needed brightness limits yet. Read their maximum
  // only when shown; using threshold + 1 here would make them look saturated.
  return channel.high === undefined && !channel.visible ? undefined : Math.max(Number(channel.high) || 0, low + 1);
}

function syncResolveRanges(cells) {
  state.layers.filter((l) => l.type === "image" && l.token === cells.image_token).forEach((image) => {
    image.channels.forEach((channel) => {
      const marker = cells.live.markers.find((m) => m.marker === channel.name);
      if (marker && channel.resolveLinked !== false) {
        channel.low = marker.threshold;
        channel.high = markerHigh(channel, channel.low);
      }
    });
    remember(image); refreshLayer(image);
  });
}

function autoQuery(layer, channel, options) {
  const params = new URLSearchParams({ ch: channel.index, method: options.method || "Triangle" });
  Object.entries({...options.parameters || thresholdOptions(layer), ...(options.where || regionParams(options.scope || "", layer.token))}).forEach(([key, value]) => params.set(key, value));
  return "/image/" + layer.token + "/background?" + params;
}

function setMarkerUpper(layer,channel,high){
  // Upper-only adjustment must never invoke linked RESOLVE threshold updates.
  if(!Number.isFinite(high)||high<=Number(channel.low??0))return false;
  delete layer.thresholdPreview;channel.high=high;
  const input=layer.inputs?.[channel.index]?.high;if(input)input.value=high;
  remember(layer);refreshLayer(layer);layer.redrawHistogram?.();return true;
}

async function runMarkerAuto(layer,channels,mode='lower'){
  if(layer.autoRun)return;
  const run={cancelled:false,done:0,total:channels.length,text:'Measuring…'},applied=[],failures=[];
  const options={method:layer.autoMethod||'Triangle',scope:layer.autoScope||'',parameters:thresholdOptions(layer)};
  const percentile=layer.displayAuto?.upper_percentile??99.8;
  layer.autoRun=run;layer.autoUndo=null;buildPanel();
  try{
    options.where=regionParams(options.scope,layer.token);
    for(const channel of channels){
      if(run.cancelled||!state.layers.includes(layer))break;
      run.text='Measuring '+channel.name+' ('+run.done+'/'+run.total+')';layer.autoProgress?.();
      try{
        let path=autoQuery(layer,channel,options);
        if(mode!=='lower')path=path.replace('/background?','/display_range?')+'&'+new URLSearchParams({mode,low:channel.low??0,upper_percentile:percentile});
        const result=await getJSON(path);if(run.cancelled||!state.layers.includes(layer))break;
        const previous={channel,low:channel.low,high:channel.high,upperOnly:mode==='upper'};
        const changed=mode==='upper'?setMarkerUpper(layer,channel,result.high):setMarkerRange(layer,channel,mode==='both'?result.low:result.threshold,mode==='both'?result.high:markerHigh(channel,result.threshold));
        if(!changed)throw Error('Invalid display range.');applied.push(previous);
        run.text=(mode==='upper'?'Upper P'+percentile:options.method+(mode==='both'?' + upper P'+percentile:''))+': '+applied.length+'/'+run.total+' markers · '+result.pixels.toLocaleString()+' pixels · 1/'+number(result.downsample)+' resolution';
      }catch(error){failures.push(channel.name+': '+error.message);}
      run.done++;layer.autoProgress?.();
    }
    layer.autoMessage=run.cancelled?'Cancelled after '+applied.length+' markers. Undo restores their ranges.':run.text;
    if(failures.length)layer.autoMessage=(applied.length+' updated; '+failures.length+' unchanged. '+failures.join(' '));
  }catch(error){layer.autoMessage='Stopped after '+applied.length+' markers: '+error.message;message(layer.autoMessage,true);}
  finally{layer.autoUndo=applied;layer.autoRun=null;if(state.layers.includes(layer))buildPanel();}
}

function thresholdOptions(layer) {
  return {percentile:layer.autoParameters?.percentile ?? 95, manual:layer.autoParameters?.manual ?? 0, native:layer.autoNative ? 1 : 0};
}

function syncThresholdParameterWidgets(layer) {
  document.querySelectorAll('[data-threshold-token="'+CSS.escape(layer.token)+'"]').forEach(input=>{
    if(input===document.activeElement)return;
    const key=input.dataset.thresholdParameter;
    input.value=key==='native'?(layer.autoNative?'native':'preview'):(layer.autoParameters?.[key] ?? (key==='percentile'?95:0));
  });
  for(const observer of layer.histogramObservers||[]){
    if(observer.box.isConnected)observer.sync();else layer.histogramObservers.delete(observer);
  }
}

function thresholdParameterControls(layer, method) {
  const box=el('div',{class:'inline threshold-parameters'});
  const parameters=layer.autoParameters ||= {percentile:95,manual:0};
  for(const [key,label] of [['percentile','Percentile'],['manual','Manual cutoff']]) {
    const input=el('input',{type:'number',value:parameters[key] ?? (key==='percentile'?95:0),step:'any','aria-label':label});
    if(key==='percentile'){input.min=0;input.max=100;}
    input.disabled=!!layer.autoRun; input.required=true;input.dataset.thresholdToken=layer.token;input.dataset.thresholdParameter=key;
    input.onchange=()=>{if(!input.checkValidity()||!Number.isFinite(Number(input.value))){input.value=parameters[key]??(key==='percentile'?95:0);message('Enter a finite cutoff, or a percentile from 0 to 100.',true);return;}parameters[key]=Number(input.value);remember(layer);syncThresholdParameterWidgets(layer);};
    const row=field(label,input);row.hidden=method!==(key==='percentile'?'Percentile':'Manual');box.append(row);
  }
  const resolution=select([['preview','Preview resolution'],['native','Full resolution']],layer.autoNative?'native':'preview',value=>{layer.autoNative=value==='native';remember(layer);syncThresholdParameterWidgets(layer);});
  resolution.setAttribute('aria-label','Threshold sampling resolution'); resolution.disabled=!!layer.autoRun;resolution.dataset.thresholdToken=layer.token;resolution.dataset.thresholdParameter='native';
  box.append(resolution,help('Preview uses the image pyramid for speed. Full resolution streams original integer pixels in bounded windows. The benchmark reports the actual resolution. All Auto controls use this same choice.'));
  return box;
}

function thresholdAnnotations(layer) {
  const visible=state.layers.filter(l=>l.type==='annotations'&&l.image_token===layer.token);
  return visible.concat((layer.thresholdAnnotations||[]).filter(a=>!visible.some(b=>a.token===b.token)));
}

function autoControls(layer, {displayRange=false}={}) {
  const box = el("div", { class: "auto-controls", "data-threshold-image":layer.token });
  const method = select((layer.info.auto_methods || ["Triangle"]).map((m) => [m, m]), layer.autoMethod || "Triangle",
    (v) => {
      layer.autoMethod = v; remember(layer);
      // Keep both copies synchronized without rebuilding annotation choices.
      document.querySelectorAll('[data-threshold-image="'+CSS.escape(layer.token)+'"]').forEach(control=>{
        control.querySelector('[aria-label="Auto method"]').value=v;
        control.querySelector('.threshold-parameters')?.replaceWith(thresholdParameterControls(layer,v));
      });
    });
  method.disabled = !!layer.autoRun;
  const regions = regionOptions(layer.token);
  const scopes = [["", "Whole image"]].concat(regions.filter(([key]) => key !== null));
  const scope = regionSelect([["", "Whole image"]], regions, layer.autoScope || "", (v) => {
    layer.autoScope = v; remember(layer); loadSelectedClasses();
  });
  scope.disabled = !!layer.autoRun;
  // Keep an unavailable selection explicit; never silently measure all pixels.
  if (layer.autoScope && !scopes.some(([key]) => key === layer.autoScope)) {
    let chosen = {};
    try { chosen = JSON.parse(layer.autoScope); } catch (e) { /* a shape */ }
    const source = state.layers.find((l) => l.type === "annotations" && l.token === chosen.annotation);
    scope.appendChild(el("option", { value: layer.autoScope, text: source ? source.name + ": " + (chosen.class || "all areas") : "Saved annotation choice" }));
    scope.value = layer.autoScope;
  }
  const annotations = thresholdAnnotations(layer);
  const loadedClasses = new Set();
  async function loadClasses(annotation) {
    if (loadedClasses.has(annotation.token)) return;
    loadedClasses.add(annotation.token);
    const add = (value, name) => {
      const existing = Array.from(scope.options).find((o) => o.value === value);
      if (existing) existing.textContent = name;
      else scope.appendChild(el("option", { value: value, text: name }));
      scope.value = layer.autoScope || "";
    };
    try {
      const legend = await annotationLegend(annotation);
      legend.items.forEach((item) => add(JSON.stringify({ annotation: annotation.token, class: item.value }), annotation.name + ": " + item.value));
    } catch (error) {
      loadedClasses.delete(annotation.token);
      message("Could not list annotation classes: " + error.message, true);
    }
  }
  function loadSelectedClasses() {
    let selected = {};
    try { selected = JSON.parse(layer.autoScope || "{}"); } catch (e) { /* a shape or whole image */ }
    annotations.filter((a) => a.token === selected.annotation).forEach(loadClasses);
  }
  annotations.forEach((annotation) => {
    scope.appendChild(el("option", { value: JSON.stringify({ annotation: annotation.token }), text: annotation.name + ": all areas" }));
    // Reuse classes already requested by the user, without reading hidden files.
    if (annotation.legendRequest) loadClasses(annotation);
  });
  scope.value = layer.autoScope || "";
  scope.addEventListener("focus", loadSelectedClasses);
  scope.addEventListener("pointerdown", loadSelectedClasses);
  method.setAttribute("aria-label", "Auto method");
  scope.setAttribute("aria-label", "Measure within");
  box.appendChild(field("Auto", el("div", { class: "inline" }, [method, el("span", { class: "muted", text: "in" }), scope]),
    "Auto sets each marker's min to its background level, measured on the image's pixels with the chosen method " +
    "(the Background step's methods). Choose where to measure: a drawn layer counts all its polygons together (a tissue " +
    "in many pieces is measured once), a name counts every polygon with that name, and Selected polygons counts the ones " +
    "picked on the image (click, Ctrl-click to add). Annotation files list their classes when you open the list. " +
    "RESOLVE thresholds use the cells' saved measurements."));
  box.append(thresholdParameterControls(layer, layer.autoMethod || "Triangle"));
  const all = el("button", { class: "btn", text: "Auto all markers", title: displayRange ? "Set the lower cutoff and upper display level using the options below" : "Set lower background cutoffs", onclick: () => runMarkerAuto(layer, layer.channels.slice(), displayRange ? (layer.displayAuto?.adjust || "both") : "lower") });
  all.disabled = !!layer.autoRun;
  const undoAuto = el("button", { class: "btn", text: "Undo Auto", title: "Restore the ranges from before the last Auto", onclick: () => {
    (layer.autoUndo || []).forEach(({ channel, low, high, upperOnly }) => {
      if(upperOnly){channel.high=high;return;}
      if (Number.isFinite(low)) setMarkerRange(layer, channel, low,
        high === undefined && channel.visible ? markerHigh(channel, low) : high);
      else { channel.low = low; channel.high = high; }
    });
    layer.autoUndo = null; layer.autoMessage = "Previous min and max restored.";
    remember(layer); refreshLayer(layer); buildPanel();
  } });
  undoAuto.disabled = !!layer.autoRun || !(layer.autoUndo || []).length;
  const buttons = el("div", { class: "row wrap" }, [all, undoAuto]);
  if (layer.autoRun) buttons.appendChild(el("button", { class: "btn", text: "Cancel", onclick: () => {
    layer.autoRun.cancelled = true; layer.autoRun.text = "Stopping after this marker..."; layer.autoProgress();
  } }));
  if(displayRange){
    const upper=el('button',{class:'btn',text:'Auto upper',title:'Adjust only the upper display level of all markers; lower cutoffs stay unchanged',onclick:()=>runMarkerAuto(layer,layer.channels.slice(),'upper')});
    upper.disabled=!!layer.autoRun;buttons.insertBefore(upper,undoAuto);
    const settings=layer.displayAuto||={adjust:'both',upper_percentile:99.8};
    const summary=el('summary',{text:'Display range · '+(settings.adjust==='both'?'low + upper':settings.adjust==='upper'?'upper only':'low only')+' · upper P'+settings.upper_percentile});
    const refreshSummary=()=>{summary.textContent='Display range · '+(settings.adjust==='both'?'low + upper':settings.adjust==='upper'?'upper only':'low only')+' · upper P'+settings.upper_percentile;remember(layer);};
    const adjust=select([['both','Lower + upper'],['lower','Lower only'],['upper','Upper only']],settings.adjust,v=>{settings.adjust=v;refreshSummary();});adjust.setAttribute('aria-label','Auto display adjustment');adjust.disabled=!!layer.autoRun;
    const percentile=el('input',{type:'number',value:settings.upper_percentile,min:.001,max:100,step:'any',required:true,'aria-label':'Upper display percentile'});percentile.disabled=!!layer.autoRun;
    percentile.onchange=()=>{if(!percentile.checkValidity()){percentile.value=settings.upper_percentile;return;}settings.upper_percentile=Number(percentile.value);refreshSummary();};
    box.append(el('details',{class:'display-auto-options'},[summary,field('Auto adjusts',adjust),field('Upper %',percentile),el('p',{class:'hint',text:'Upper = this percentile of pixels above the lower cutoff, in the selected area. 99.8 leaves only the brightest 0.2% at full colour. Use 100 to retain the maximum. This upper level changes visualization only.'})]));
  }
  box.appendChild(buttons);
  const status = el("div", { class: "hint", role: "status", "aria-live": "polite" });
  // The image card and Studio's Background step can both show these controls.
  const statuses = layer.autoStatuses || (layer.autoStatuses = new Set());
  statuses.add(status);
  layer.autoProgress = () => {
    statuses.forEach((box) => {
      if (!box.isConnected && box !== status) { statuses.delete(box); return; }
      box.textContent = layer.autoRun ? layer.autoRun.text : layer.autoMessage || "";
      box.hidden = !box.textContent;
    });
    if (layer.syncLevels) layer.syncLevels();
  };
  layer.autoProgress(); box.appendChild(status);
  return box;
}

// An "All" checkbox for the top of a list: ticked when every item is on,
// half-ticked when some are; ticking turns every item on, unticking all off.
function allBox(text, onChange) {
  const box = el("input", { type: "checkbox", title: "Tick to select all, untick to clear all" });
  const row = el("label", { class: "all-row" }, [box, el("span", { text: text })]);
  box.addEventListener("click", (e) => e.stopPropagation());
  box.addEventListener("change", () => onChange(box.checked));
  return {
    row: row,
    sync: (states) => {
      const on = states.filter(Boolean).length;
      box.checked = states.length > 0 && on === states.length;
      box.indeterminate = on > 0 && on < states.length;
    },
  };
}


async function importAnnotationCopy(payload) {
  const image = state.layers.find(l=>l.type==='image');
  if (!image) return;
  const session = new URLSearchParams(location.search).get('session');
  try {
    const result = await postJSON('/studio/'+encodeURIComponent(session)+'/annotation_sources', {action:'import',image_token:image.token,...payload});
    if (!state.layers.includes(image)) return;
    const layer = await addLayerToView({...result.layer,open:true});
    state.drawInto=layer.id;(state.layerGroups||(state.layerGroups={})).annotations=true;
    state.layerGroups['annotation-imported']=true;
    buildPanel();updateToolbar();message('Imported '+layer.features.length+' editable polygons.');
  } catch(error) {message(error.message,true);}
}

function tissueImportDialog(initial) {
  const image=state.layers.find(l=>l.type==='image');
  const session=new URLSearchParams(location.search).get('session');
  if(!image)return;
  $('tissue-import-dialog')?.close();
  const candidates=state.layers.filter(l=>l.image_token===image.token&&(l.type==='annotations'||l.annotation_group==='imported'));
  const dialog=el('dialog',{id:'tissue-import-dialog',class:'annotation-import-dialog','aria-labelledby':'tissue-import-title'});
  const source=el('select',{'aria-label':'Tissue import source'},[
    el('option',{value:'',text:'Choose a loaded layer…',disabled:''}),
    ...candidates.map(l=>el('option',{value:l.token,text:l.name,title:l.saved_file||l.name}))]);
  const names=el('select',{'aria-label':'Tissue import annotation',disabled:''});
  const status=el('span',{class:'muted',role:'status'});
  const use=el('button',{class:'btn primary',text:'Use as Tissue',disabled:''});
  let found=null,sequence=0;
  const active=()=>dialog.isConnected&&state.layers.includes(image);
  const field=(text,input)=>el('div',{class:'row'},[el('label',{class:'lbl',text}),input]);
  const load=async()=>{
    const request=++sequence;found=null;use.disabled=true;names.disabled=true;names.replaceChildren();status.textContent='Loading annotations…';
    try{
      await flushAllShapes();
      const result=await getJSON('/studio/'+encodeURIComponent(session)+'/annotation_sources?source_token='+encodeURIComponent(source.value)+'&image_token='+encodeURIComponent(image.token));
      if(!active()||request!==sequence)return;
      found=result;
      names.replaceChildren(el('option',{value:'choose',text:'Choose an annotation…',disabled:''}),
        ...result.classes.map(c=>el('option',{value:JSON.stringify(c.name),text:(c.name||'Unclassified')+' · '+c.count+' polygons'})),
        el('option',{value:'null',text:'All polygons · '+result.count}));
      const preferred=result.classes.find(c=>/^(tissue|valid_tissue)$/i.test(c.name))||(result.classes.length===1?result.classes[0]:null);
      names.value=preferred?JSON.stringify(preferred.name):'choose';names.disabled=!!result.read_only;
      use.disabled=result.read_only||!result.count||names.value==='choose';
      status.textContent=result.count?'': 'No polygon annotations in this layer.';
    }catch(error){if(active()&&request===sequence)status.textContent=error.message;}
  };
  source.onchange=load;
  names.onchange=()=>{use.disabled=!found||found.read_only||names.value==='choose';};
  use.onclick=async()=>{
    if(!found||!active())return;
    const tissue=state.layers.find(l=>l.id==='tissue-regions');
    if(tissue?.preview&&!window.confirm('Replace the current unsaved Tissue preview with this annotation?'))return;
    use.disabled=true;source.disabled=true;names.disabled=true;status.textContent='Opening in Tissue…';
    try{
      await flushAllShapes();
      const result=await postJSON('/studio/'+encodeURIComponent(session)+'/annotation_sources',{
        action:'use_as_tissue',image_token:image.token,source_token:source.value,source_revision:found.revision,class_name:JSON.parse(names.value)});
      if(!active())return;
      if(!await state.studio.useImportedTissue(result))throw Error('The image changed. Open its annotations again.');
      dialog.close();message(result.features.length+' polygons opened in Tissue. Save polygons to keep them.');
    }catch(error){if(dialog.isConnected)status.textContent=error.message;}
    finally{if(active()){source.disabled=false;names.disabled=!!found?.read_only;use.disabled=!!found?.read_only;}}
  };
  dialog.append(el('div',{class:'row'},[el('h3',{id:'tissue-import-title',text:'Import tissue'}),
    help('Choose a loaded GeoJSON layer and the annotation class to use as Tissue. A copy opens for editing; the source file stays intact. Save polygons keeps it as the original Tissue, ready for artifact removal. No detection or area filtering is applied.')]),
    field('Source',source),field('Annotation',names),status,el('div',{class:'row'},[use,
      el('button',{class:'btn',text:'Cancel',onclick:()=>dialog.close()})]));
  dialog.addEventListener('close',()=>dialog.remove());document.body.append(dialog);dialog.showModal();
  source.value=initial?.token||(candidates.length===1?candidates[0].token:'');
  if(source.value)load();else if(!candidates.length)status.textContent='Load a GeoJSON using Import annotations in Layers first.';
}
function annotationImportControls() {
  const file=el('input',{type:'file',accept:'.geojson,.json',hidden:true});
  file.addEventListener('change',async()=>{
    if(!file.files[0])return;
    const item=file.files[0];
    try {
      const raw=JSON.parse(await item.text());
      const features=Array.isArray(raw)?raw:raw.type==='FeatureCollection'?raw.features:raw.type==='Feature'?[raw]:null;
      if(!features)throw Error('Choose a GeoJSON polygon file.');
      await importAnnotationCopy({name:item.name,features});
    }catch(error){message(error.message,true);}
  });
  return el('div',{class:'row'},[file,el('button',{class:'btn small',text:'Import annotations',onclick:()=>file.click()}),
    help('Import GeoJSON from QuPath or another app in this image’s pixel coordinates. A separate editable copy is saved in this project. Use Work within on the left to select it for analysis; visibility only controls its outline.')]);
}
