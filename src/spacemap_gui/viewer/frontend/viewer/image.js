/* SPACEMAP viewer, part 3 of 12: the image itself, saving it as a picture, and Studio's figures.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- viewer
function createViewer() {
  const viewer = OpenSeadragon({
    element: $("osd"),
    showNavigationControl: false,
    showNavigator: true,
    navigatorPosition: "BOTTOM_RIGHT",
    navigatorSizeRatio: 0.16,
    drawer: "canvas",
    // OSD compares screen pixels per source pixel, including display density.
    // 0.5 selects a sufficiently fine level; 0.9 enlarged coarse tiles below 90%.
    minPixelRatio: 0.5,
    imageSmoothingEnabled: false,
    // Changing a marker, colour or range replaces the layer; without this the
    // viewer zooms out to the whole image every time the image is the only
    // layer shown. The first view is set in placeFirstView() instead.
    preserveViewport: true,
    crossOriginPolicy: false,
    maxZoomPixelRatio: 6,
    minZoomImageRatio: 0.6,
    visibilityRatio: 0.3,
    blendTime: 0.1,
    immediateRender: false,
    timeout: 120000,
    imageLoaderLimit: 8,
    gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
    gestureSettingsTouch: { clickToZoom: false },
  });
  state.viewer = viewer;
  // Registration: with Move H&E on, dragging moves the H&E over the CODEX instead of the view.
  viewer.addHandler("canvas-drag", (event) => {
    const he = state.alignMove && state.layers.find((l) => l.align && l.item);
    if (!he) return;
    event.preventDefaultAction = true;
    const unit = alignUnit(he);
    if (!unit) return;
    const delta = viewer.viewport.deltaPointsFromPixels(event.delta, true);
    he.align.x += delta.x / unit; he.align.y += delta.y / unit;
    applyAlign(he);
  });
  state.imagesReady = false;
  state.cellsReady = false;
  state.whenImagesReady = new Promise((resolve) => { state.releaseImageWait = resolve; });
  state.whenCellsReady = new Promise((resolve) => { state.releaseCellWait = resolve; });
  viewer.addHandler("tile-load-failed", (event) => {
    const layer = state.layers.find((l) => l.item === event.tiledImage);
    if (layer) layer.loadFailed = true;
    message("An image tile could not load. Try reopening the image.", true);
  });
  openLayersInOrder(viewer);
  ensureOverlay();
  renderShapes();
  ["update-viewport", "open", "resize", "add-item"].forEach((name) => viewer.addHandler(name, placeOverlay));
  viewer.addHandler("canvas-double-click", (event) => {
    if (state.tool === "polygon") { event.preventDefaultAction = true; finishPolygon(); }
  });
  document.querySelectorAll("#drawtools button").forEach((button) => {
    button.onclick = () => {
      if (button.getAttribute("data-mode") === "select") {  // selection mode: outlines select regions instead of drawing
        state.selectMode = !state.selectMode;
        if (state.selectMode && !["rect", "polygon", "freehand"].includes(state.tool)) setTool("rect"); else updateToolbar();
        message(state.selectMode ? "Select: draw around regions to select them (Shift adds); right-click one to name them." : "Drawing again.");
        return;
      }
      const tool = button.getAttribute('data-tool'), target = toolbarDrawTarget();
      if (tool !== 'off' && !state.selectMode && target?.type === 'shapes' && !target.visible) setVisible(target, true);
      setTool(button.getAttribute("data-tool"));
    };
  });
  updateToolbar();

  $("btn-home").onclick = () => viewer.viewport.goHome();
  $("btn-home").title = "Fit image (or click the mouse wheel)";
  $("btn-native").onclick = () => zoomImagePercent(100);
  // Wheel-click is navigation only, even while a drawing/brush tool is active.
  // Cancel mousedown to prevent the browser's automatic scrolling mode.
  viewer.canvas.addEventListener("mousedown", event => {
    if (event.button === 1) { event.preventDefault(); event.stopImmediatePropagation(); }
  }, true);
  viewer.canvas.addEventListener("auxclick", event => {
    if (event.button !== 1) return;
    event.preventDefault(); event.stopImmediatePropagation();
    viewer.viewport.goHome();
  }, true);
  $("btn-zoom-in").onclick = () => { viewer.viewport.zoomBy(1.6); viewer.viewport.applyConstraints(); };
  $("btn-zoom-out").onclick = () => { viewer.viewport.zoomBy(1 / 1.6); viewer.viewport.applyConstraints(); };
  $("btn-save").onclick = saveImage;
  $("btn-save-full").onclick = saveFullResolution;
  $("btn-window").hidden = STANDALONE || !state.spec.session;
  $("btn-minimize").hidden = STANDALONE;
  $("btn-window").onclick = () => {
    // Opened from the click itself, so pop-up blockers allow it.
    // The window listens on this browser's channel; SPACEMAP publishes the view
    // there as soon as it hears the id (it comes with this event).
    const channel = browserId() || state.spec.session;
    const url = api("/window/index.html?session=" + encodeURIComponent(channel));
    const opened = window.open(url, "spacemap-viewer-" + channel, "popup,width=1500,height=950");
    if (!opened) { message("The browser blocked the window. Allow pop-ups for SPACEMAP and try again.", true); return; }
    saveView();
    setComponentValue({ event: "window", browser: browserId(), n: Date.now() });
  };
  $("btn-minimize").onclick = () => { saveView(); setComponentValue({ event: "minimize", browser: browserId(), n: Date.now() }); };
  $("btn-fullscreen").onclick = async () => {
    if (document.fullscreenElement) { document.exitFullscreen(); return; }
    try {
      await document.documentElement.requestFullscreen();
    } catch (error) {
      // The pop-up inside SPACEMAP cannot go full screen; a window of its own can.
      message(STANDALONE ? "The browser did not allow full screen." : "Use Separate window, then Full screen there.", true);
    }
  };
  viewer.addHandler("animation-finish", saveView);
  state.placedFirstView = false;

  viewer.addHandler("animation", updateScaleBar);
  viewer.addHandler("open", updateScaleBar);
  viewer.addHandler("resize", updateScaleBar);
  viewer.addHandler("add-item", () => setTimeout(updateScaleBar, 50));
  const tracker = new OpenSeadragon.MouseTracker({
    element: viewer.canvas,
    moveHandler: (event) => {
      moveBrushCursor(event);
      showReadout(event.position);
      followSplitMarkerCursor(event.position);
      if (state.tool === "polygon" && drawing.points) { drawing.hover = toImage(event.position); drawPreview(); }
    },
  });
  tracker.setTracking(true);
  viewer.addHandler("canvas-click", onClick);
  viewer.addHandler("canvas-press", onPress);
  viewer.addHandler("canvas-drag", onDrag);
  viewer.addHandler("canvas-release", onRelease);
  viewer.addHandler("canvas-scroll", brushScroll);
  viewer.addHandler("canvas-exit", () => { state.brushPointer = null; updateBrushCursor(); });
}

async function openLayersInOrder(viewer) {
  const layers = state.layers.slice();
  const waitFor = async (group) => {
    while (state.viewer === viewer && group.some((layer) => layer.visible && !layer.loadFailed &&
      (!layer.item || !layer.item.getFullyLoaded()))) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  };
  const images = layers.filter((layer) => layer.visible && layer.type === "image");
  images.forEach(addLayer);
  await waitFor(images);
  if (state.viewer !== viewer) return;
  state.imagesReady = true;
  state.releaseImageWait();
  if (state.studio) refreshNamePalette();  // the project's names, for this image's right-click menu
  const cells = layers.filter((layer) => layer.visible && layer.type === "cells");
  cells.forEach(addLayer);
  await waitFor(cells);
  if (state.viewer !== viewer) return;
  state.cellsReady = true;
  state.releaseCellWait();
  // Optional overlays load one at a time, after the image and cells.
  for (const layer of layers.filter((l) => l.type === "annotations" || l.type === "density")) {
    if (state.viewer !== viewer) return;
    if (layer.visible) { addLayer(layer); await waitFor([layer]); }
  }
}

// The first layer to open sets the view once: where you left this image, or
// the whole image. Later layer changes never move it.
function placeFirstView() {
  if (state.placedFirstView || !state.viewer) return;
  state.placedFirstView = true;
  const viewport = state.viewer.viewport;
  const saved = state.restoreViewport;
  if (saved && saved.center) {
    viewport.zoomTo(saved.zoom, null, true);
    viewport.panTo(new OpenSeadragon.Point(saved.center[0], saved.center[1]), true);
  } else {
    viewport.goHome(true);
  }
  updateScaleBar();
}

// OpenSeadragon's overview map (navigator) mirrors every reorder of the main image's layers. A layer swapped in a
// moment ago may not have its mirror yet; the library then inserts an empty slot (it only warns) and the next frame
// fails on it. Ignore a reorder of a layer that is not (yet) there; its mirror is placed when it is created.
(function guardSetItemIndex() {
  const world = OpenSeadragon.World.prototype, setItemIndex = world.setItemIndex;
  if (setItemIndex.guarded) return;
  world.setItemIndex = function (item, index) {
    if (!item || this.getIndexOfItem(item) < 0 || !(index >= 0 && index < this._items.length)) return;
    return setItemIndex.call(this, item, index);
  };
  world.setItemIndex.guarded = true;
})();

// Layers open asynchronously, so each is put back in panel order once it has opened.
function restack() {
  let index = 0;
  for (const layer of state.layers) {
    if (!layer.item || state.viewer.world.getIndexOfItem(layer.item) < 0) continue;
    if (state.viewer.world.getIndexOfItem(layer.item) !== index) state.viewer.world.setItemIndex(layer.item, index);
    index += 1;
  }
}

function addLayer(layer) {
  if (layer.adding || layer.item) return;
  const gate = layer.type === "cells" && !state.imagesReady ? state.whenImagesReady :
    (layer.type === "annotations" || layer.type === "density") && !state.cellsReady ? state.whenCellsReady : null;
  if (gate) {
    gate.then(() => { if (layer.visible && state.layers.includes(layer)) addLayer(layer); });
    return;
  }
  const viewer = state.viewer;
  layer.adding = true;
  layer.loadFailed = false;
  viewer.addTiledImage({
    tileSource: tileSource(layer),
    opacity: layer.opacity,
    success: (event) => {
      layer.adding = false;
      if (state.viewer !== viewer) return;
      if (!layer.visible) { state.viewer.world.removeItem(event.item); return; }  // hidden while opening
      layer.item = event.item;
      applySwipe(layer);
      state.layers.filter((l) => l.align).forEach(applyAlign);
      restack();
      placeFirstView();
    },
    error: () => {
      layer.adding = false; layer.loadFailed = true;
      if (state.viewer === viewer) message("Could not draw the layer " + (layer.name || layer.id) + ".", true);
    },
  });
}

// A changed layer (a marker, colour, range or column) is loaded invisibly on
// top of the old one and swapped in only when its tiles are all there, so the
// image never flashes blank. A newer change cancels an older one still loading.
function retire(layer) {
  const world = state.viewer.world;
  (layer.retiring || []).forEach((old) => { if (world.getIndexOfItem(old) >= 0) world.removeItem(old); });
  layer.retiring = [];
}

function refreshLayer(layer) {
  if (layer.type === 'image') syncSplitMarkers();
  const viewer = state.viewer;
  if (layer.type === "shapes") { renderShapes(); return; }
  if (!layer.visible) return;
  debounce("refresh-" + layer.id, () => {
    if (state.viewer !== viewer || !state.layers.includes(layer)) return;
    if (layer.adding) { refreshLayer(layer); return; }  // try again once it has opened
    if (!layer.item) { addLayer(layer); return; }
    const world = state.viewer.world;
    const previous = layer.item;
    if (!(layer.retiring || []).includes(previous)) layer.retiring = (layer.retiring || []).concat([previous]);
    const generation = (layer.generation || 0) + 1;
    layer.generation = generation;
    state.viewer.addTiledImage({
      tileSource: tileSource(layer), index: world.getIndexOfItem(previous) + 1, opacity: 0, preload: true,
      success: (event) => {
        const fresh = event.item;
        if (state.viewer !== viewer) return;
        if (!layer.visible || generation !== layer.generation) { world.removeItem(fresh); return; }
        layer.item = fresh;
        applySwipe(layer);
        applyAlign(layer);
        let swapped = false;
        const swap = () => {
          if (state.viewer !== viewer) return;
          if (swapped || generation !== layer.generation || layer.item !== fresh || !layer.visible || !state.layers.includes(layer)) return;
          swapped = true;
          fresh.setOpacity(layer.opacity);
          layer.retiring = (layer.retiring || []).filter((old) => old !== fresh);
          retire(layer);
          restack();
        };
        // Loading events can fire inside OpenSeadragon's item update loop.
        // Change the stack after that loop returns, not while it walks its items.
        if (fresh.getFullyLoaded()) setTimeout(swap, 0);
        else {
          fresh.addHandler("fully-loaded-change", (e) => { if (e.fullyLoaded) setTimeout(swap, 0); });
          setTimeout(swap, 5000);  // never wait for ever on a slow tile
        }
      },
    });
  }, 150);
}

function setVisible(layer, visible) {
  layer.visible = visible;
  saveView();
  if (layer.type === "shapes") { renderShapes(); updateToolbar(); return; }
  if (!visible) retire(layer);
  if (visible && !layer.item) addLayer(layer);
  if (!visible && layer.item) { state.viewer.world.removeItem(layer.item); layer.item = null; }
}

function setOpacity(layer, opacity) {
  layer.opacity = opacity;
  saveView();
  if (layer.type === "shapes") { renderShapes(); return; }
  if (layer.item) layer.item.setOpacity(opacity);
}

function baseItem() {
  const layer = state.layers.find((l) => l.item);
  return layer ? layer.item : null;
}

function toImage(position) {
  const item = baseItem();
  if (!item) return null;
  const point = state.viewer.viewport.pointFromPixel(position);
  return item.viewportToImageCoordinates(point);
}

function pixelSize() { return (state.base && state.base.info.pixel_size_um) || null; }

function imageZoom() {
  const item=state.base?.item||state.layers.find(l=>l.type==='image'&&l.item)?.item;
  return item&&state.viewer ? item.viewportToImageZoom(state.viewer.viewport.getZoom(true)) : null;
}

function zoomImagePercent(percent, point=null) {
  const item=state.base?.item||state.layers.find(l=>l.type==='image'&&l.item)?.item;
  if(!item||!state.viewer)return;
  state.viewer.viewport.zoomTo(item.imageToViewportZoom(percent/100),point);
}

function updateImageSampling() {
  const zoom=imageZoom();if(!Number.isFinite(zoom)||zoom<=0)return;
  // Filter reductions to avoid aliasing, but show enlarged source pixels without
  // the extra blur of bilinear interpolation. The optional preference is per image.
  const smooth=!!state.pixelSmoothing || zoom*(OpenSeadragon.pixelDensityRatio||1)<1-1e-6;
  if(state.viewer.displaySmoothing!==smooth){
    state.viewer.displaySmoothing=smooth;
    state.viewer.drawer.setImageSmoothingEnabled(smooth);
  }
  const native=$("btn-native");
  if(native){native.title='Current zoom: '+(zoom*100).toFixed(1)+'%. Click for 100%: one image pixel per viewer pixel. Higher zoom enlarges the existing pixels.';
    native.classList.toggle('on',Math.abs(zoom-1)<.001);}
}

function showReadout(position) {
  const point = toImage(position);
  if (!point) return;
  const size = pixelSize();
  let text = "x " + Math.round(point.x) + "  y " + Math.round(point.y) + " px";
  if (size) text += "   (" + (point.x * size).toFixed(1) + ", " + (point.y * size).toFixed(1) + " µm)";
  $("readout").textContent = text;
}

function niceLength(target) {
  const power = Math.pow(10, Math.floor(Math.log10(target)));
  for (const step of [1, 2, 5, 10]) if (step * power >= target) return step * power;
  return 10 * power;
}

function scaleBarModel() {
  const item = baseItem();
  const size = pixelSize();
  if (!item || !size) return null;
  const zoom = item.viewportToImageZoom(state.viewer.viewport.getZoom(true)); // screen px per image px
  const micronsPerPixel = size / zoom;
  const length = niceLength(micronsPerPixel * 120);
  return { width: length / micronsPerPixel, label: length >= 1000 ? (length / 1000) + " mm" : length + " µm" };
}

function updateScaleBar() {
  updateImageSampling();
  const model = scaleBarModel();
  $("scalebar").hidden = !model;
  if (!model) return;
  $("scalebar").querySelector(".bar").style.width = model.width + "px";
  $("scalebar").querySelector(".text").textContent = model.label;
}

// ---------------------------------------------------------------- export
function drawScaleBar(context, x, y, width, label, ratio) {
  context.fillStyle = "#fff"; context.strokeStyle = "#000"; context.lineWidth = 1 * ratio;
  context.fillRect(x, y, width, 6 * ratio); context.strokeRect(x, y, width, 6 * ratio);
  context.font = (13 * ratio) + "px sans-serif"; context.textAlign = "center";
  context.lineWidth = 3 * ratio; context.strokeText(label, x + width / 2, y + 22 * ratio);
  context.fillText(label, x + width / 2, y + 22 * ratio);
}

function download(canvas, suffix) {
  try {
    canvas.toBlob((blob) => {
      if (!blob) { message("The browser could not create the image.", true); return; }
      const link = document.createElement("a");
      const base = (state.spec.title || "spacemap_view").replace(/[^A-Za-z0-9_.-]+/g, "_");
      link.download = base + suffix + ".png";
      link.href = URL.createObjectURL(blob);
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 5000);
    }, "image/png");
  } catch (error) {
    message("Could not save the image: " + error.message, true);
  }
}

function saveImage() {
  const canvas = viewCanvas(true);
  if (canvas) download(canvas, "_view");
}

// The view as drawn on screen (image, cells, shapes), optionally with its scale bar.
function viewCanvas(withScaleBar) {
  const source = state.viewer && state.viewer.drawer && state.viewer.drawer.canvas;
  if (!source) { message("Nothing to save yet.", true); return null; }
  const canvas = document.createElement("canvas");
  canvas.width = source.width; canvas.height = source.height;
  const context = canvas.getContext("2d");
  context.fillStyle = "#000"; context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(source, 0, 0);
  const ratio = source.width / Math.max(1, source.clientWidth || source.width);
  const item = baseItem();
  if (item) {
    drawShapesOnCanvas(context, (x, y) => {
      const p = item.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));
      return [p.x * ratio, p.y * ratio];
    }, ratio);
  }
  const model = scaleBarModel();
  if (model && withScaleBar) drawScaleBar(context, 16 * ratio, canvas.height - 34 * ratio, model.width * ratio, model.label, ratio);
  return canvas;
}

const EXPORT_MAX = 8192;  // longest side of a full-resolution export, in pixels

async function saveFullResolution() {
  const button = $("btn-save-full");
  button.disabled = true;
  try {
    const found = await fullCanvas(true);
    if (found) download(found.canvas, found.suffix);
  } catch (error) {
    message("Could not save at full resolution: " + error.message, true);
  } finally {
    button.disabled = false;
  }
}

// The visible area read from the tiles at the finest level that fits EXPORT_MAX, with the shapes.
async function fullCanvas(withScaleBar) {
  const item = baseItem();
  if (!item || !state.base) { message("Nothing to save yet.", true); return null; }
  const info = state.base.info;
  const rect = item.viewportToImageRectangle(state.viewer.viewport.getBounds(true));
  const x0 = Math.max(0, Math.floor(rect.x)), y0 = Math.max(0, Math.floor(rect.y));
  const x1 = Math.min(info.width, Math.ceil(rect.x + rect.width)), y1 = Math.min(info.height, Math.ceil(rect.y + rect.height));
  if (x1 <= x0 || y1 <= y0) { message("The image is not on screen.", true); return null; }
  let z = 0;
  while (Math.max(x1 - x0, y1 - y0) / Math.pow(2, z) > EXPORT_MAX && z < info.max_zoom) z += 1;
  const scale = Math.pow(2, z), span = info.tile_size * scale;
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil((x1 - x0) / scale); canvas.height = Math.ceil((y1 - y0) / scale);
  const context = canvas.getContext("2d");
  context.fillStyle = "#000"; context.fillRect(0, 0, canvas.width, canvas.height);
  const tiles = [];
  for (let ty = Math.floor(y0 / span); ty <= Math.floor((y1 - 1) / span); ty++)
    for (let tx = Math.floor(x0 / span); tx <= Math.floor((x1 - 1) / span); tx++) tiles.push([tx, ty]);
  const layers = state.layers.filter((l) => l.visible && l.item);
  {
    for (const [number, layer] of layers.entries()) {
      const version = state.versions[layer.id] || 0;
      let done = 0;
      const images = new Array(tiles.length);
      const next = { i: 0 };
      const worker = async () => {
        while (next.i < tiles.length) {
          const index = next.i++;
          const [tx, ty] = tiles[index];
          const response = await fetch(tileUrl(layer, z, tx, ty, version, true), { credentials: "same-origin" });
          if (!response.ok) throw new Error("tile " + tx + "," + ty + ": HTTP " + response.status);
          images[index] = await createImageBitmap(await response.blob());
          done += 1;
          message("Saving at full resolution: layer " + (number + 1) + " of " + layers.length + ", " +
            Math.round(100 * done / tiles.length) + "%");
        }
      };
      await Promise.all(Array.from({ length: 6 }, worker));
      context.globalAlpha = layer.opacity;
      tiles.forEach(([tx, ty], index) => {
        context.drawImage(images[index], (tx * span - x0) / scale, (ty * span - y0) / scale);
        images[index].close();
      });
    }
    context.globalAlpha = 1;
    drawShapesOnCanvas(context, (x, y) => [(x - x0) / scale, (y - y0) / scale], Math.max(1, canvas.width / 1200));
    const size = pixelSize();
    if (size && withScaleBar) {
      const micronsPerPixel = size * scale;
      const length = niceLength(micronsPerPixel * canvas.width / 8);
      const ratio = Math.max(1, canvas.width / 1200);
      drawScaleBar(context, 16 * ratio, canvas.height - 34 * ratio, length / micronsPerPixel,
        length >= 1000 ? (length / 1000) + " mm" : length + " µm", ratio);
    }
    message("Saved " + canvas.width + " x " + canvas.height + " px" + (z ? " (1/" + scale + " of full resolution, the largest that fits)" : "") + ".");
    return {canvas, suffix:"_x" + x0 + "_y" + y0 + (z ? "_1of" + scale : "_full")};
  }
}

// ---------------------------------------------------------------- figure (Studio's Visualization step)
// What the colours mean: the visible markers, then the cells' colour column (a list, or a gradient).
async function figureLegend() {
  const entries = [];
  const image = state.layers.find((l) => l.type === "image" && l.visible);
  const markers = image ? (image.channels || []).filter((c) => c.visible) : [];
  if (markers.length) {
    entries.push({ kind: "heading", label: "Markers" });
    markers.forEach((c) => entries.push({ kind: "swatch", label: c.label || c.name, color: c.color }));
  }
  const cells = state.layers.find((l) => l.type === "cells" && l.visible && l.color_by && !l.live);
  if (cells) {
    try {
      const legend = await getJSON("/cells/" + cells.token + "/legend?t=" + encodeURIComponent(cells.table || "") +
        "&col=" + encodeURIComponent(cells.color_by) + "&cmap=" + encodeURIComponent(cells.colormap || "viridis") +
        "&hide=" + encodeURIComponent(Array.from(cells.hidden || []).join("\u001f")));
      entries.push({ kind: "heading", label: cells.color_by });
      if (legend.kind === "numeric") entries.push({ kind: "gradient", colormap: legend.colormap, min: legend.min, max: legend.max });
      else (legend.items || []).filter((item) => !(cells.hidden || new Set()).has(item.value)).slice(0, 40)
        .forEach((item) => entries.push({ kind: "swatch", label: String(item.value), color: item.color }));
    } catch (error) { /* a figure without the cell legend is still a figure */ }
  }
  return entries;
}

function drawFigure(base, title, entries) {
  const ratio = Math.max(1, base.width / 1200);
  const top = title ? Math.round(44 * ratio) : 0, side = entries.length ? Math.round(230 * ratio) : 0;
  const row = Math.round(20 * ratio);
  const height = Math.max(base.height + top, top + Math.round(16 * ratio) + entries.length * row);
  const canvas = document.createElement("canvas");
  canvas.width = base.width + side; canvas.height = height;
  const context = canvas.getContext("2d");
  context.fillStyle = "#000"; context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(base, 0, top);
  context.fillStyle = "#fff"; context.textBaseline = "middle"; context.textAlign = "left";
  if (title) { context.font = "600 " + Math.round(22 * ratio) + "px sans-serif"; context.fillText(title, 16 * ratio, top / 2); }
  let y = top + Math.round(16 * ratio);
  const x = base.width + Math.round(14 * ratio), box = Math.round(12 * ratio);
  entries.forEach((entry) => {
    context.font = (entry.kind === "heading" ? "600 " : "") + Math.round(13 * ratio) + "px sans-serif";
    context.fillStyle = "#fff";
    if (entry.kind === "heading") context.fillText(entry.label, x, y + row / 2);
    else if (entry.kind === "swatch") {
      context.fillStyle = entry.color; context.fillRect(x, y + (row - box) / 2, box, box);
      context.fillStyle = "#fff"; context.fillText(entry.label, x + box + 8 * ratio, y + row / 2);
    } else {
      const stops = gradientStops(entry.colormap).split(","), width = side - 28 * ratio;
      const gradient = context.createLinearGradient(x, 0, x + width, 0);
      stops.forEach((colour, index) => gradient.addColorStop(index / (stops.length - 1), colour));
      context.fillStyle = gradient; context.fillRect(x, y + 2 * ratio, width, box);
      context.fillStyle = "#fff"; context.font = Math.round(11 * ratio) + "px sans-serif";
      context.fillText(Number(entry.min).toFixed(1), x, y + row + 2 * ratio);
      context.textAlign = "right"; context.fillText(Number(entry.max).toFixed(1), x + width, y + row + 2 * ratio); context.textAlign = "left";
      y += row;
    }
    y += row;
  });
  return canvas;
}

// A figure: the view (or the area at full resolution), a title, the legend and the scale bar; downloaded as PNG.
async function saveFigure({ title = "", legend = true, scaleBar = true, full = false } = {}) {
  const found = full ? await fullCanvas(scaleBar) : { canvas: viewCanvas(scaleBar), suffix: "_view" };
  if (!found || !found.canvas) return null;
  const figure = drawFigure(found.canvas, String(title || "").slice(0, 200), legend ? await figureLegend() : []);
  download(figure, "_figure");
  return { width: figure.width, height: figure.height };
}

// Registration: the H&E over the CODEX, placed by the pair's pixel sizes, turned and moved by hand (layer.align:
// codex_mpp, he_mpp, scaled, rotation in degrees, x and y offsets in CODEX pixels).
function alignBase(layer) { return state.layers.find((l) => l.type === "image" && l !== layer && !l.align && l.item); }
function alignUnit(layer) {
  const base = alignBase(layer);
  return base ? base.item.getBounds(true).width / base.item.getContentSize().x : null;  // world units per CODEX pixel
}
function applyAlign(layer) {
  const a = layer.align;
  if (!a || !layer.item) return;
  const base = alignBase(layer), unit = alignUnit(layer);
  if (!base || !unit) return;
  const bounds = base.item.getBounds(true);
  const heMpp = a.he_mpp || layer.info.pixel_size_um, codexMpp = a.codex_mpp || base.info.pixel_size_um;
  const scale = (a.scaled && heMpp && codexMpp ? heMpp / codexMpp : 1) * (a.zoom || 1);  // CODEX pixels per H&E pixel
  layer.item.setWidth(layer.item.getContentSize().x * scale * unit, true);
  layer.item.setPosition(new OpenSeadragon.Point(bounds.x + a.x * unit, bounds.y + a.y * unit), true);
  if (layer.item.setRotation) layer.item.setRotation(a.rotation || 0, true);
}

// Right click on the image (Move tool): zoom to a magnification, fit, centre, copy the position, the H&E.
function closeContextMenu() { const menu = $("ctxmenu"); if (menu) { menu.beforeClose?.(); menu.remove(); } }
document.addEventListener("contextmenu", (event) => {
  if (event.target.closest?.('#split-marker-panel, #split-marker-toggle')) return;
  const stage = $("stage");
  if (!state.viewer || !stage || !stage.contains(event.target)) return;
  const stageBounds = state.viewer.element.getBoundingClientRect();
  const neckPoint = toImage(new OpenSeadragon.Point(event.clientX - stageBounds.left, event.clientY - stageBounds.top));
  if (rightClickShape(neckPoint)) return;
  if (["freehand", "polygon", "rect", "merge", "split"].includes(state.tool) && !state.selectMode && !neckAt(neckPoint)) return;
  event.preventDefault();
  event.stopPropagation();
  closeContextMenu();
  const bounds = state.viewer.element.getBoundingClientRect();
  const pixel = new OpenSeadragon.Point(event.clientX - bounds.left, event.clientY - bounds.top);
  const point = state.viewer.viewport.pointFromPixel(pixel);
  const base = state.base && state.base.item ? state.base : state.layers.find((l) => l.type === "image" && l.item);
  const mpp = pixelSize() || (base && base.align && base.align.codex_mpp);
  const act = (text, run, disabled) => el("button", { type: "button", text, disabled: disabled ? "" : null,
    onclick: () => { closeContextMenu(); run(); } });
  const he = state.layers.find((l) => l.id === "he" && l.type === "image");
  const imagePoint = base && base.item ? base.item.viewportToImageCoordinates(point) : null;
  const items = [
    cellPhenotypeMenu(),
    ...(neckAt(imagePoint) ? neckMenuItems(imagePoint, act) : regionMenuItems(imagePoint, act)),
    el("div", { class: "ctx-title", text: 'Image zoom · '+((imageZoom()||0)*100).toFixed(0)+'%' }),
    el("div", { class: "ctx-row" }, [25,50,100,200,400].map(p=>act(p===100?'100% (native)':p+'%',()=>zoomImagePercent(p,point),!base))),
    act("Fit the whole image", () => state.viewer.viewport.goHome()),
    el('label',{class:'ctx-sampling'},[el('input',{type:'checkbox',checked:state.pixelSmoothing?'':null,'aria-label':'Smooth enlarged pixels',
      onchange:event=>{state.pixelSmoothing=event.target.checked;updateImageSampling();saveView();}}),
      el('span',{text:'Smooth enlarged pixels',title:'Display preference only. Off shows the source pixel boundaries when enlarged; on blends them. Saved for this image.'})]),
    act("Centre here", () => state.viewer.viewport.panTo(point)),
    act("Copy this position (µm)", () => {
      const image = base.item.viewportToImageCoordinates(point);
      const text = mpp ? (image.x * mpp).toFixed(1) + ", " + (image.y * mpp).toFixed(1) + " µm" : Math.round(image.x) + ", " + Math.round(image.y) + " px";
      try { navigator.clipboard.writeText(text); } catch (_) {}
      message("Position " + text + " copied.");
    }, !base),
    he ? act(he.visible ? "Hide the H&E" : "Show the H&E", () => setVisible(he, !he.visible)) : null,
    act("Save a picture of the view", () => $("btn-save") && $("btn-save").click()),
  ].filter(Boolean);
  const menu = el("div", { id: "ctxmenu", role: "menu" }, items);
  if(menu.querySelector('.cell-phenotype-menu'))menu.classList.add('has-cell-phenotypes');
  menu.style.left = Math.min(event.clientX, window.innerWidth - 230) + "px";
  menu.style.top = Math.min(event.clientY, window.innerHeight - 260) + "px";
  document.body.appendChild(menu);
  menu.style.top=Math.max(8,Math.min(event.clientY,window.innerHeight-menu.getBoundingClientRect().height-8))+'px';
  menu.reposition=()=>{
    menu.style.left=Math.max(8,Math.min(event.clientX,window.innerWidth-menu.getBoundingClientRect().width-8))+'px';
    menu.style.top=Math.max(8,Math.min(event.clientY,window.innerHeight-menu.getBoundingClientRect().height-8))+'px';
  };
  menu.reposition();
}, true);
document.addEventListener("pointerdown", (event) => { if (!event.target.closest || !event.target.closest("#ctxmenu")) closeContextMenu(); }, true);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeContextMenu();
  // Registration: arrow keys move the H&E while Move H&E is on (Shift: ten times further).
  const he = state.alignMove && state.layers.find((l) => l.align && l.item);
  if (!he || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key) || event.target.closest?.("input, select, textarea")) return;
  event.preventDefault();
  const step = event.shiftKey ? 100 : 10;
  he.align.x += event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0;
  he.align.y += event.key === "ArrowUp" ? -step : event.key === "ArrowDown" ? step : 0;
  applyAlign(he);
});
