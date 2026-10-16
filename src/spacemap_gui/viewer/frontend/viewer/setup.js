/* SPACEMAP viewer, part 2 of 12: opening a view: set-up, the remembered view, live refresh, tile sources.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- set-up
async function init(spec) {
  setUpDrop();  // files dropped on the image (Studio)
  const generation = state.initGeneration = (state.initGeneration || 0) + 1;
  saveView();
  await flushViewerSettings();
  if (labelSaves.queue.length || labelSaves.busy) {
    await flushLabels();
    if (labelSaves.queue.length || labelSaves.busy) {
      message("Labels still need saving. Use Save now before changing images.", true); return;
    }
  }
  if (generation !== state.initGeneration) return;
  state.viewReady = false;
  resetSplitMarkers();
  if (state.releaseImageWait) state.releaseImageWait();
  if (state.releaseCellWait) state.releaseCellWait();
  if (state.spec?.id !== spec.id) state.tissueLayerId = "tissue-regions";
  state.spec = spec;
  state.studio?.viewChanged(spec);
  state.layers = [];
  clearInterval(state.pendingTimer);
  clearInterval(state.tableTimer);
  state.undo = [];
  state.selected = null;
  state.inspectedCell = null;
  state.cellInspectionSelection = null;
  state.cellInspectEpoch = (state.cellInspectEpoch || 0) + 1;
  state.inspectedAnnotation = null;
  state.annotationProfiles = new WeakMap();
  state.cardSeq += 1;
  drawing.points = null; drawing.start = null;
  if (state.svgRoot) state.svgRoot.innerHTML = "";
  $("title").textContent = spec.title || "";
  $("panel").innerHTML = "";
  $("cellcard").hidden = true;
  if (state.viewer) { state.viewer.destroy(); state.viewer = null; }
  const saved = await loadProjectView(spec);
  if (generation !== state.initGeneration) return;
  state.removedLayers = new Set(saved?.removed || []);
  const extras = (saved && saved.extras) || [];
  const layers = [];
  try {
    // The server owns workflow layers. Remembered extras must never add a
    // second copy or replace their current editing/file metadata.
    for (const source of uniqueViewLayers(spec.layers || [], extras)) {
      if (viewRemoved(source)) continue;
      const layer = { ...source };
      const token = layer.type === "image" ? layer.token : layer.image_token;
      if (!state.infos[token]) state.infos[token] = await getJSON("/image/" + token + "/info");
      if (generation !== state.initGeneration) return;
      layer.info = state.infos[token];
      const quietDefault = spec.layers?.some(l => l.default_dapi);
      // Quiet first open: only the image, except the layers the step itself opened (its tissue, ROIs, regions), which
      // are what the step is about. A remembered view still decides afterwards (applySavedView).
      layer.visible = quietDefault && !source.added ? layer.id === spec.layers.find(l => l.type === 'image')?.id : layer.visible !== false;
      layer.opacity = layer.opacity === undefined ? 1 : layer.opacity;
      layer.hidden = new Set(layer.hidden || []);
      layer.hiddenNames = new Set(layer.hiddenNames || []);
      if (layer.type === "image" && !layer.info.rgb) await prepareChannels(layer, savedLayer(saved, layer));
      if (layer.type === "shapes") await loadShapes(layer);
      if (generation !== state.initGeneration) return;
      layers.push(layer);
    }
  } catch (error) {
    if (generation !== state.initGeneration) return;
    message("Could not open the image: " + error.message, true);
    return;
  }
  state.layers = layers;
  restoreLabelDrafts();
  applySavedView(saved);
  if (spec.note) message(spec.note);
  state.base = state.layers.find((l) => l.type === "image") || state.layers[0];
  createViewer();
  state.viewReady = true;
  buildPanel();
  watchTables();
  watchPendingCells();
  saveView();  // Rewrite older remembered views without duplicate layer entries.
  state.studio?.imageReady?.(spec);
  state.phenotypingLink = dataRefreshCombined().catch(error => message('Cell table: ' + error.message, true));
}

function uniqueViewLayers(primary, extras = []) {
  const seen = new Set();
  return primary.concat(extras).filter(layer => {
    if (layer.id === 'clean-tissue-regions') return false; // retained in Tissue History instead
    if (seen.has(layer.id)) return false;
    seen.add(layer.id); return true;
  });
}

// Cell outlines prepared in the background: draw the cells once they are ready.
function watchPendingCells() {
  clearInterval(state.pendingTimer);
  const pending = () => state.layers.filter((l) => l.type === "cells" && l.pending);
  if (!pending().length) return;
  state.pendingTimer = setInterval(async () => {
    for (const layer of pending()) {
      try {
        const result = await getJSON("/cells/" + layer.token + "/ready");
        if (!result.ready) continue;
        layer.pending = false;
        bump(layer);
        message(result.cells === 0 ? "This segmentation has no numbered cells to show." :
          "Cell outlines ready (" + Number(result.cells || 0).toLocaleString() + " cells).");
      } catch (error) { /* try again */ }
    }
    if (!pending().length) clearInterval(state.pendingTimer);
  }, 2000);
}

// ---------------------------------------------------------------- remembered view
// The view (position, zoom, layer settings and order, added layers) is kept
// per image, so Minimize and Restore, or reopening, come back as you left it.
const VIEW_FIELDS = ["visible", "opacity", "mode", "color_by", "table", "colormap", "outline", "open", "selection",
  "fcol", "fvals", "vcol", "radius", "weighted", "cmap", "cmapAuto", "profileSource", "suggestionsOn", "learnColumn", "suggestionMode", "suggestionModelId", "markerTransforms", "activeSet", "colourRows", "range", "filters"];

function loadView(id) {
  try { return JSON.parse(localStorage.getItem("spacemap-viewer-view:" + id) || "null"); } catch (e) { return null; }
}

// Studio's own annotation layers (Marker regions, Classification, copies, imports) are deleted in the project, not
// hidden in this view: a result made again (the same label run again) must show, whatever an older view remembered.
function viewRemoved(layer) {
  if (!state.removedLayers?.has(viewLayerKey(layer))) return false;
  return !(state.studio && ["regions", "classes", "imported"].includes(layer.annotation_group));
}
function viewLayerKey(layer) {
  // Numbered imported layers can change order when files are added to a project.
  return layer.type === 'annotations' ? 'annotation:' + layer.token + (layer.added ? ':' + layer.id : '') : layer.id;
}
function savedLayer(saved, layer) {
  return saved?.layers?.[saved.keyedLayers ? viewLayerKey(layer) : layer.id];
}
function viewStorageKey(spec) { return 'spacemap-viewer-project-view:' + spec.workspace; }

async function loadProjectView(spec) {
  state.viewPersistenceBlocked = false;
  if (spec.workspace) {
    // A timed-out save may still be queued (or already written without a reply).
    // Returning to this image must not replace newer local edits with old JSON.
    const pending = pendingViewerSetting('/workspace/' + spec.workspace + '/view');
    if (pending?.view) return pending.view;
    try {
      const result = await getJSON('/workspace/' + spec.workspace + '/view');
      if (result.view) return result.view;
    } catch (error) {
      // Never overwrite unreadable project settings with defaults.
      state.viewPersistenceBlocked = true;
      message('Could not load the saved view: ' + error.message, true);
    }
    try {
      const cached = JSON.parse(localStorage.getItem(viewStorageKey(spec)) || 'null');
      if (cached) return cached;
    } catch (e) { /* browser storage is optional */ }
  }
  return loadView(spec.id);  // migrate an existing browser-only view once
}

const viewerSettingWrites = new Map();
const viewerWriter = Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);
let viewerWriteSequence = 0;
let viewerSaveNotice = null;
function pendingViewerSetting(path) {
  const body = viewerSettingWrites.get(api(path))?.pending;
  return body ? JSON.parse(body) : null;
}
function viewerSavingNotice() {
  const waiting = Array.from(viewerSettingWrites.values()).filter(job => job.pending && job.failure);
  const rejected = waiting.find(job => !job.failure.transient);
  const prolonged = waiting.some(job => job.failures >= 3);
  const text = rejected ? 'Viewer settings could not be saved: ' + rejected.failure.message :
    prolonged ? 'Viewer settings are still waiting for confirmation. Retrying in the background.' :
    waiting.length ? 'Saving viewer settings is taking longer. Retrying…' : null;
  if (text && text !== viewerSaveNotice) message(text, !!rejected || prolonged);
  if (!text && viewerSaveNotice && $('message').textContent === viewerSaveNotice) {
    clearTimeout(messageTimer); $('message').hidden = true;
  }
  viewerSaveNotice = text;
}
function queueViewerSetting(path, body) {
  const key = api(path);
  let job = viewerSettingWrites.get(key);
  if (!job) { job = {key, pending:null, running:null, last:null, failures:0, failure:null}; viewerSettingWrites.set(key, job); }
  const content = JSON.stringify(body);
  if (job.last === content) return;
  job.last = content;
  job.pending = JSON.stringify({...body, stamp:{writer:viewerWriter, sequence:++viewerWriteSequence}});
  clearTimeout(job.timer);
  job.timer = setTimeout(() => writeViewerSetting(job), 500);
}
async function writeViewerSetting(job) {
  if (job.running) return job.running;
  if (!job.pending) return;
  clearTimeout(job.timer);
  job.running = (async () => {
    while (job.pending) {
      const body = job.pending;
      const controller = new AbortController(), timeout = setTimeout(() => controller.abort(),
        Math.min(30000, 10000 * (1 + job.failures)));
      try {
        const response = await fetch(job.key, {method:'POST', credentials:'same-origin',
          headers:{'Content-Type':'application/json'}, body, signal:controller.signal});
        if (!response.ok) {
          // Proxies can return HTML on a temporary 502/503. Keep the HTTP status
          // so that an unavailable server is not reported as an invalid setting.
          const result = await response.json().catch(() => ({}));
          const error = Error(result.error || ('HTTP ' + response.status)); error.status = response.status; throw error;
        }
        await response.json();
        if (job.pending === body) job.pending = null;
        job.failures = 0; job.failure = null; viewerSavingNotice();
      } catch (error) {
        job.failures += 1;
        job.failure = {message:error.message, transient:controller.signal.aborted || error.name === 'AbortError' ||
          error instanceof TypeError || [408, 425, 429, 500, 502, 503, 504].includes(error.status)};
        viewerSavingNotice();
        clearTimeout(job.timer);
        job.timer = setTimeout(() => writeViewerSetting(job), Math.min(30000, 5000 * job.failures));
        break;
      } finally { clearTimeout(timeout); }
    }
  })();
  try { await job.running; } finally { job.running = null; }
}
async function flushViewerSettings() {
  await Promise.all(Array.from(viewerSettingWrites.values(), job => { clearTimeout(job.timer); return writeViewerSetting(job); }));
}
function keepViewerSettings() {
  saveView();
  for (const job of viewerSettingWrites.values()) {
    if (!job.pending) continue;
    // A sequence stamp also protects against an older in-flight save arriving last.
    fetch(job.key, {method:'POST', credentials:'same-origin', keepalive:true,
      headers:{'Content-Type':'application/json'}, body:job.pending}).catch(() => {});
  }
}
window.addEventListener('pagehide', keepViewerSettings);
document.addEventListener('visibilitychange', () => { if (document.hidden) keepViewerSettings(); });

function saveView() {
  if (!state.spec || !state.viewer || !state.viewReady) return;
  const layers = {};
  state.layers.forEach((layer) => {
    const kept = {};
    VIEW_FIELDS.forEach((field) => { if (layer[field] !== undefined) kept[field] = layer[field]; });
    if (layer.channels) {
      kept.channels = channelSettings(layer);
      kept.autoMethod = layer.autoMethod; kept.autoScope = layer.autoScope; kept.autoParameters=layer.autoParameters; kept.autoNative=layer.autoNative; kept.displayAuto=layer.displayAuto; kept.histogramView=layer.histogramView;
    }
    kept.hidden = Array.from(layer.hidden || []);
    kept.hiddenNames = Array.from(layer.hiddenNames || []);
    layers[viewLayerKey(layer)] = kept;
  });
  const owned = new Set((state.spec.layers || []).map(l => l.id));
  const extras = uniqueViewLayers(state.layers).filter((l) => l.added && !owned.has(l.id)).map((l) => {
    const copy = {};
    ["id", "type", "name", "token", "image_token", "tables", "added", "editable", "regions", "annotation_kind", "artifact_part", "training_part", "tissue_version", "annotation_group", "classes", "default_class", "names", "saved_file", "origin", "source_name", "source", "imported_cells", "data_layer", "data_text", "data_show", "data_size", "data_reach", "data_within", "overlay", "hidden_tables"].concat(VIEW_FIELDS).forEach((f) => { if (l[f] !== undefined) copy[f] = l[f]; });
    copy.hidden = Array.from(l.hidden || []);
    copy.hiddenNames = Array.from(l.hiddenNames || []);
    if (l.preview) copy.editable = l.preview.editable;
    return copy;
  });
  const viewport = state.viewer.viewport;
  const center = viewport.getCenter(true);
  const position = !state.placedFirstView && state.restoreViewport?.center ? state.restoreViewport :
    {center:[center.x, center.y], zoom:viewport.getZoom(true)};
  const record = { layers: layers, order: state.layers.map(viewLayerKey), extras: extras, keyedLayers:true,
    removed:Array.from(state.removedLayers || []),
    panelLayout:1, layerGroups:state.layerGroups || {}, shownOnly:!!state.layersShownOnly,
    center: position.center, zoom: position.zoom, pixelSmoothing:!!state.pixelSmoothing,
    splitMarkers:{...splitMarkerOptions()} };
  try { localStorage.setItem("spacemap-viewer-view:" + state.spec.id, JSON.stringify(record)); } catch (e) { /* storage full or blocked */ }
  if (state.spec.workspace) {
    try { localStorage.setItem(viewStorageKey(state.spec), JSON.stringify(record)); } catch (e) { /* optional */ }
    if (!state.viewPersistenceBlocked) queueViewerSetting('/workspace/' + state.spec.workspace + '/view', {view:record});
  }
}

function applySavedView(saved) {
  Object.assign(splitMarkerOptions(), splitMarkerPreferences(saved?.splitMarkers || splitMarkerOptions()));
  state.pixelSmoothing = saved?.pixelSmoothing === true;
  state.layerGroups = {...saved?.layerGroups};
  state.layersShownOnly = !!saved?.shownOnly;
  if (!saved?.panelLayout) state.layers.forEach(layer => { layer.open = layer.type === 'image'; });
  if (!saved) return;
  state.layers.forEach((layer) => {
    const kept = savedLayer(saved, layer);
    if (!kept) return;
    VIEW_FIELDS.forEach((field) => { if (kept[field] !== undefined && (field !== 'open' || saved.panelLayout)) layer[field] = kept[field]; });
    layer.hidden = new Set(kept.hidden || []);
    layer.hiddenNames = new Set(kept.hiddenNames || []);
  });
  const order = saved.order || [];
  state.layers.sort((a, b) => {
    const i = order.indexOf(saved.keyedLayers ? viewLayerKey(a) : a.id), j = order.indexOf(saved.keyedLayers ? viewLayerKey(b) : b.id);
    return (i < 0 ? 1e6 : i) - (j < 0 ? 1e6 : j);
  });
  state.restoreViewport = saved.center ? saved : null;
}

// ---------------------------------------------------------------- live refresh
// When a tab writes new columns or labels into a table, cells coloured from
// it are redrawn, here and in a separate viewer window.
function watchTables() {
  clearInterval(state.tableTimer);
  state.stamps = {};
  state.tableTimer = setInterval(async () => {
    const seen = new Set();
    for (const layer of state.layers) {
      if (layer.type !== "cells" || !layer.table || seen.has(layer.token + layer.table)) continue;
      seen.add(layer.token + layer.table);
      try {
        const table = layer.table;
        const result = await getJSON("/cells/" + layer.token + "/stamp?t=" + encodeURIComponent(table));
        if (!state.layers.includes(layer) || table !== layer.table) continue;
        const key = layer.token + "|" + table;
        if (state.stamps[key] !== undefined && state.stamps[key] !== result.stamp) {
          state.layers.filter((l) => l.token === layer.token && (l.type === "cells" || l.type === "density")).forEach((l) => {
            l.columns = null;
            labelSaves.changed.add(l);
          });
          refreshLabelColours();
          if (dataView.holder?.isConnected && (dataView.source?.combined || dataLinkedLayer() === layer)) dataLoad(true);
        }
        state.stamps[key] = result.stamp;
      } catch (error) { /* try again next time */ }
    }
  }, 5000);
}

async function prepareChannels(layer, viewSettings) {
  const names = layer.info.channels || [];
  const given = {};
  (layer.channels || []).forEach((c) => { given[c.index] = c; });
  const stored = {};
  if (layer.display_token) {
    try {
      const result = pendingViewerSetting('/display/' + layer.display_token + '/save') ||
        await getJSON("/display/" + layer.display_token);
      layer.autoMethod = (result.auto || {}).method || "Triangle";
      layer.autoScope = (result.auto || {}).scope || ""; layer.autoParameters=(result.auto || {}).parameters; layer.autoNative=!!result.auto?.native; layer.displayAuto=result.auto?.display; layer.histogramView=result.auto?.histogram;
      (result.channels || []).forEach((c) => { if (names[c.index] === c.name) stored[c.index] = c; });
    } catch (error) { /* fall back to the presets */ }
  }
  if (!Object.keys(stored).length && viewSettings?.channels) {
    viewSettings.channels.forEach(c => { if (names[c.index] === c.name) stored[c.index] = c; });
    layer.autoMethod = viewSettings.autoMethod || 'Triangle'; layer.autoScope = viewSettings.autoScope || ''; layer.autoParameters=viewSettings.autoParameters; layer.autoNative=!!viewSettings.autoNative; layer.displayAuto=viewSettings.displayAuto; layer.histogramView=viewSettings.histogramView;
  }
  if (!layer.display_token && !viewSettings?.channels) {
    try {
      const options = JSON.parse(localStorage.getItem(storageKey(layer) + ":auto") || "{}");
      layer.autoMethod = options.method || "Triangle"; layer.autoScope = options.scope || ""; layer.autoParameters=options.parameters; layer.autoNative=!!options.native; layer.displayAuto=options.display; layer.histogramView=options.histogram;
    } catch (e) { /* use defaults */ }
  }
  const saved = Object.keys(stored).length ? null : recall(layer);
  // The quiet first-open default shows the DAPI channel (DAPI-01, DAPI_2… too); with none, the first channel.
  const dapi = names.findIndex(name => /dapi/i.test(String(name)));
  const firstShown = dapi >= 0 ? dapi : 0;
  layer.channels = names.map((name, index) => {
    const preset = given[index] || {};
    const channel = {
      index: index, name: name, label: name,
      visible: layer.default_dapi ? index === firstShown : !!preset.visible,
      // The image file's colour (as QuPath shows it), else a preset, else the palette; saved choices below win.
      color: (layer.info.colors || [])[index] || preset.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length],
      low: preset.low, high: preset.high,
    };
    const kept = stored[index];
    const fileColour = (layer.info.colors || [])[index];
    if (kept) {
      // A saved colour counts when the user chose it (or the file records none); otherwise the file's colour is used.
      channel.visible = kept.visible; channel.label = kept.label || name;
      if (kept.custom || !fileColour) { channel.color = kept.color; channel.custom = !!kept.custom; }
      if (kept.low !== null && kept.low !== undefined) channel.low = kept.low;
      if (kept.high !== null && kept.high !== undefined) channel.high = kept.high;
    } else if (saved && saved[index]) {
      channel.visible = saved[index][0];
      if (saved[index][4] || !fileColour) { channel.color = saved[index][1]; channel.custom = !!saved[index][4]; }
      if (saved[index][2] !== null && saved[index][2] !== undefined) channel.low = saved[index][2];
      if (saved[index][3] !== null && saved[index][3] !== undefined) channel.high = saved[index][3];
    }
    return channel;
  });
  // Explicit all-off is a saved choice. Only non-catalog custom views need a
  // first-channel fallback when no preset or remembered choice exists.
  if (!Object.keys(stored).length && !saved && !layer.default_dapi && !Object.keys(given).length && layer.channels.length) layer.channels[0].visible = true;
  await fillLimits(layer, layer.channels.filter((c) => c.visible && (c.low === undefined || c.high === undefined)));
}

async function fillLimits(layer, channels) {
  if (!channels.length) return;
  const limits = await getJSON("/image/" + layer.token + "/limits?ch=" + channels.map((c) => c.index).join(","));
  channels.forEach((c) => {
    const pair = limits[String(c.index)];
    if (pair) {
      if (c.low === undefined) c.low = Math.round(pair[0]);
      if (c.high === undefined) c.high = Math.max(Math.round(pair[1]), c.low + 1);
    }
  });
}

// ---------------------------------------------------------------- tile sources
function tileSource(layer) {
  const info = layer.info;
  const zmax = info.max_zoom;
  const version = state.versions[layer.id] || 0;
  // Image tiles are RGB even when encoded as PNG. OSD otherwise infers alpha
  // from the extension and clears fractional tile edges, leaving hairline gaps.
  // Checkerboards and overlay tiles do contain alpha and must keep that path.
  const transparent = layer.type !== "image" || !!layer.checker;
  return {
    width: info.width, height: info.height, tileSize: info.tile_size, tileOverlap: 0,
    minLevel: 0, maxLevel: zmax,
    hasTransparency: function () { return transparent; },
    getTileUrl: function (level, x, y) { return tileUrl(layer, zmax - level, x, y, version); },
  };
}

function tileUrl(layer, z, x, y, version, lossless = false) {
  const where = "/" + z + "/" + x + "/" + y;
  if (layer.type === "image") {
    // Native detail and figure exports retain every displayed RGB value. Zoomed-out
    // overviews remain compact; a render version avoids reusing older JPEG tiles.
    const format = z === 0 || lossless ? 'png' : 'jpg';
    let url = api("/image/" + layer.token + "/tile" + where + "." + format + "?render=2&v=" + encodeURIComponent(layer.info.fingerprint));
    if (layer.thresholdPreview) url += "&threshold_preview=" + encodeURIComponent([layer.thresholdPreview.channel,layer.thresholdPreview.threshold,layer.thresholdPreview.colour,layer.thresholdPreview.mode].join(":"));
    if (layer.checker) url += "&ck=" + layer.checker;  // every other square see-through (Registration's checkerboard)
    if (!layer.info.rgb) {
      // "Before" (Studio's Background step) shows the markers as measured: every minimum at 0, the levels kept.
      const style = layer.channels.filter((c) => c.visible)
        .map((c) => [c.index, hex(c.color), layer.showBefore ? 0 : (c.low || 0), c.high || 1].join(":")).join(";");
      url += "&s=" + encodeURIComponent(style);
    }
    return url;
  }
  if (layer.type === "cells" && layer.live) {  // coloured by live RESOLVE's phenotypes
    return api("/resolve/" + layer.resolve + "/tile" + where + ".png?mode=" + (layer.mode === "outline" ? "fill" : (layer.mode || "fill")) +
      "&ol=" + hex(layer.outline || "#ffff00") + "&hide=" + encodeURIComponent(Array.from(layer.liveHidden || []).join("\u001f")) +
      "&near=" + (layer.nearThreshold ? "1" : "0") + "&v=" + layer.live.version + "&palette=" + viewerWriter + '-' + version);
  }
  const hidden = encodeURIComponent(Array.from(layer.hidden).join("\u001f"));
  if (layer.type === "cells") {
    return api("/cells/" + layer.token + "/tile" + where + ".png?mode=" + (layer.mode || "outline") +
      "&t=" + encodeURIComponent(layer.table || "") + "&col=" + encodeURIComponent(layer.color_by || "") +
      "&cmap=" + encodeURIComponent(layer.colormap || "viridis") + "&ol=" + hex(layer.outline || "#ffff00") +
      "&selection=" + encodeURIComponent(layer.selection || '') + "&hide=" + hidden +
      (layer.range ? "&lo=" + encodeURIComponent(layer.range[0]) + "&hi=" + encodeURIComponent(layer.range[1]) : "") +
      ((layer.filters || []).length ? "&f=" + encodeURIComponent(JSON.stringify(layer.filters)) : "") + "&v=" + version + "&palette=" + viewerWriter);
  }
  if (layer.type === "density") {
    return api("/cells/" + layer.token + "/density" + where + ".png?" + densityQuery(layer) + "&v=" + version);
  }
  return api("/annotations/" + layer.token + "/tile" + where + ".png?hide=" + hidden + "&v=" + version);
}

function densityQuery(layer) {
  return "t=" + encodeURIComponent(layer.table || "") + "&fcol=" + encodeURIComponent(layer.fcol || "") +
    "&fvals=" + encodeURIComponent((layer.fvals || []).join("\u001f")) + "&vcol=" + encodeURIComponent(layer.vcol || "") +
    "&r=" + (layer.radius || 50) + "&w=" + (layer.weighted ? 1 : 0) + "&cmap=" + encodeURIComponent(layer.cmap || "inferno");
}
