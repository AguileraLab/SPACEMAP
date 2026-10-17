/* SPACEMAP Viewer: a pyramidal image viewer embedded in the SPACEMAP interface.
 *
 * Draws image layers (multichannel fluorescence or RGB), cell layers (outlines
 * or fills coloured by any per-cell column) and annotation layers as tiles
 * rendered by the SPACEMAP server, using OpenSeadragon for navigation.
 * Cells can be labelled here (Manual Reference); labels go to the fast label
 * store and are written into the CSV from the file panel. Shapes (ROIs, regions
 * to segment, background areas) are drawn on an SVG overlay and saved as GeoJSON
 * in the project; each marker's colour, range and name are saved with the
 * project, and its range can be set on a histogram as in QuPath.
 *
 * Part 1 of 12: the link to SPACEMAP, the viewer's shared state, helpers and building blocks.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- Streamlit
function send(type, data) {
  window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data || {}), "*");
}
function setFrameHeight(height) { send("streamlit:setFrameHeight", { height: height }); }
function setComponentValue(value) { send("streamlit:setComponentValue", { value: value, dataType: "json" }); }

const state = {
  spec: null, viewer: null, layers: [], infos: {}, base: null,
  tool: "off", undo: [], versions: {}, cardSeq: 0, selected: null,
};
const drawing = { points: null, start: null, hover: null, lastScreen: null };
const $ = (id) => document.getElementById(id);
const DEFAULT_COLORS = ["#4f7dff", "#00e05a", "#ff3b3b", "#ffd400", "#ff00ff", "#00e5ff", "#ff8c00",
  "#ffffff", "#9b5de5", "#7cff6b", "#ff6f91", "#2ec4b6"];

// Inside SPACEMAP the viewer is a Streamlit component; opened with ?session=
// it is a window of its own that follows the view SPACEMAP publishes.
const STANDALONE = window.parent === window;

function browserId() {
  try {
    let id = localStorage.getItem("spacemap-viewer-browser");
    if (!id) {
      const bytes = new Uint8Array(16);
      crypto.getRandomValues(bytes);
      id = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
      localStorage.setItem("spacemap-viewer-browser", id);
    }
    return id;
  } catch (error) { return null; }
}

async function followSession() {
  const session = new URLSearchParams(location.search).get("session");
  state.apiBase = location.pathname.split("/window/")[0];
  state.sessionVersion = null;
  if (new URLSearchParams(location.search).get('studio') === '1' && window.startSpacemapStudio) {
    state.studio = window.startSpacemapStudio({session, get:getJSON, post:postJSON, stream:postStream, runCommand, captureView,
      setSelectionTool:(mode,shape) => setTool(shape || 'off',mode),
      // The polygons picked on the image now (one drawn layer): for "Selected polygons" in a step.
      pickedShapes: () => { const picked = pickedShapes();
        return picked ? {shape:picked.layer.token, name:picked.layer.name, ids:Array.from(picked.ids, String)} : null; },
      refreshResolve: async () => {
        for (const layer of state.layers.filter(item => item.type === 'cells' && item.live && item.resolve)) {
          layer.live = await postJSON('/resolve/' + layer.resolve + '/open',{});
          syncResolveRanges(layer);refreshLayer(layer);
        }
        buildPanel();
      },
      renderResolve: resolveStudioPanel,
      renderBackground: backgroundStudioPanel,
      renderRois: roisStudioPanel,
      // Studio's stage: the shapes layer new drawings go into (null: the topmost editable one).
      setDrawTarget: (id) => {
        state.drawInto = id || null;
        const layer = state.layers.find(l=>l.id===id);
        if (layer && !layer.visible) { setVisible(layer, true); buildPanel(); }  // the layer a stage edits is shown
        const part = layer?.artifact_part || layer?.training_part;
        if(part && [state.selected?.layer,state.picks?.layer].some(l=>l && (l.artifact_part||l.training_part)!==part)){
          state.selected=null;state.picks=null;renderShapes();refreshRegionList();
        }
        updateToolbar();
      },
      drawingState: () => ({id:state.drawInto,tool:state.tool}),
      flushShapes: flushAllShapes,
      saveTissuePolygons: () => savePolygons(state.layers.find(l => l.id === (state.tissueLayerId || 'tissue-regions'))),
      rebuildLayers: buildPanel,
      // Spatial analysis and Visualization: draw only the cells passing *filters* (the cell layer's Filter), from the
      // first of the layer's tables holding their columns; [] shows every cell again. *colour*: colour them by it.
      filterCells: async (filters, colour) => {
        await (state.commandQueue || Promise.resolve()).catch(() => {});
        const cells = state.layers.find(item => item.type === 'cells' && (item.tables || []).length);
        const needed = [...new Set([...(filters || []).map(f => f.col), ...(colour ? [colour] : [])])];
        if (!cells || (needed.length && !(await cellTableWith(cells, needed)))) return false;
        cells.filters = filters || [];
        if (colour) { cells.live = null; cells.color_by = colour; cells.hidden = new Set(); if (cells.mode === 'outline') cells.mode = 'both'; }
        if (!cells.visible) setVisible(cells, true);
        bump(cells); buildPanel();
        return true;
      },
      // Remove small pieces (This image): the Tissue list's Remove smaller on the shown layer; Ctrl+Z brings them back.
      removeSmaller: (id, um2) => {
        const layer = state.layers.find(l => l.id === id);
        if (!layer?.features || !layer.editable) return null;
        const size = pixelSize(), limit = size ? um2 / (size * size) : um2;
        const ids = new Set(layer.features.filter(f => featureArea(f) < limit).map(f => f.properties.id));
        if (ids.size) { state.lastEdited = layer; deleteRegions(layer, ids); }
        return ids.size;
      },
      // A figure of each ticked image: wait until the image just opened has drawn its tiles (or give up after *ms*).
      imageSettled: async (ms = 30000) => {
        const start = Date.now();
        while (Date.now() - start < ms) {
          const item = baseItem();
          if (state.imagesReady && state.cellsReady !== false && item && (!item.getFullyLoaded || item.getFullyLoaded())) return true;
          await new Promise((resolve) => setTimeout(resolve, 200));
        }
        return false;
      },
      // Studio's stage letters: show the shapes layers a stage made (all their names), so a click on a green letter
      // shows that result.
      showLayersWhere: test => {
        const found = state.layers.filter(l => l.type === 'shapes' && test(l));
        for (const layer of found) { layer.hiddenNames = null; if (!layer.visible) setVisible(layer, true); }
        if (found.length) buildPanel();
        return found.length;
      },
      // Files rewritten by Studio (a rename from Export): their layers read them again.
      reloadShapeFiles: async paths => {
        const wanted = new Set(paths || []);
        for (const layer of state.layers.filter(l => l.type === 'shapes' && wanted.has(l.saved_file))) { await loadShapes(layer); refreshLayer(layer); }
        renderShapes(); buildPanel();
      },
      // Export's Imported files: every file to the image it names; those matching none wait for their image.
      importMatched: files => bringMatchedGeoJSON(Array.from(files || [])),
      unmatchedImports: () => state.unmatchedImports || [],
      forgetUnmatched: item => { state.unmatchedImports = (state.unmatchedImports || []).filter(other => other !== item); },
      showShapesLayer: async spec => { await addLayerToView({ ...spec, visible: true }); buildPanel(); },
      // Export's rules changed: Layers' Export ticks are read again.
      refreshExportTicks: () => { state.annotationExport = null; buildPanel(); },
      showTissuePreview,
      importTissue: () => tissueImportDialog(),
      activeTissueId: () => state.layers.some(l=>l.id===state.tissueLayerId) ? state.tissueLayerId : 'tissue-regions',
      chooseTissue: async (spec, reveal=true) => {
        if (!state.layers.some(l => l.type === 'image' && l.token === spec.image_token)) return false;
        const changed = state.tissueLayerId !== spec.id || state.layers.find(l=>l.id===spec.id)?.token !== spec.token;
        let layer = state.layers.find(l => l.id === spec.id);
        if (layer && layer.token !== spec.token) { removeLayer(layer); layer = null; }
        if (!layer) layer = await addLayerToView({...spec,...(!reveal?{visible:false}:{} )});
        if (!layer) return false;
        layer.working_class=spec.working_class||'';
        if(layer.working_class){layer.classes=spec.classes;layer.default_class=spec.default_class;}
        state.tissueLayerId = layer.id; state.drawInto = layer.id;
        if(reveal){layer.visible = true; layer.open = true;}
        if(changed)showNeckReview(null); renderShapes(); buildPanel(); updateToolbar(); return true;
      },
      updateWorkingTissue: async (spec, operation) => {
        if (!state.layers.some(l=>l.type==='image' && l.token===spec.image_token)) return false;
        let layer = state.layers.find(l=>l.id===spec.id && l.token===spec.token);
        if (!layer) layer = await addLayerToView({...spec});
        await flushShapeSave(layer);
        if (layer.brushBusy || layer.saveError) throw Error('Wait for Tissue edits to save first.');
        layer.brushBusy=true; shapeStatus(layer);
        try {
          const result = await operation(layer.revision);
          if(!result || !Array.isArray(result.features) || typeof result.revision!=='string')throw Error('The server did not confirm saved polygons. Restart Studio to load the current update.');
          if (!state.layers.includes(layer)) return result;
          clearShapePreview(layer,false);
          layer.history=layer.history||[];layer.history.push(JSON.stringify(result.before||layer.features||[]));layer.future=[];
          layer.features=result.features;layer.revision=result.revision;layer.historyCache=null;
          layer.status='Saved. Ctrl+Z undoes this change.';layer.visible=true;layer.saveError=null;
          state.lastEdited=layer;state.tissueLayerId=layer.id;state.selected=null;state.picks=null;
          showNeckReview(null);renderShapes();buildPanel();refreshRegionList();saveView();
          state.studio?.regionsChanged?.(layer);
          return result;
        } finally {layer.brushBusy=false;shapeStatus(layer);}
      },
      ensureArtifactLayer: async spec => {
        if (!state.layers.some(l => l.type === 'image' && l.token === spec.image_token)) return false;
        const existing=state.layers.find(l=>l.id===spec.id);
        // A layer kept from before the Output folder changed points at the old file: replace it with the current one.
        if(!existing||(spec.token&&existing.token!==spec.token&&spec.saved_file&&existing.saved_file&&spec.saved_file!==existing.saved_file))
          await addLayerToView({...spec,visible:existing?existing.visible:false});
        else for(const key of ['name','classes','artifact_part','training_part','working_class'])if(spec[key]!==undefined)existing[key]=spec[key];
        return true;
      },
      saveArtifactPolygons: part => savePolygons(state.layers.find(l => l.id === 'artifact-' + part)),
      manualArtifacts: part => (state.layers.find(l=>l.id==='artifact-'+part)?.features||[]).filter(f=>f.properties.source==='manual_artifact'),
      reviewedArtifact: part => {
        const layer = state.layers.find(l => l.id === 'artifact-' + part);
        if (!layer || layer.brushBusy || layer.saveBusy || layer.loadingShapes) throw Error('Wait for the polygons to finish loading or saving.');
        return layer.preview ? {reviewed_features:layer.features} : {reviewed_revision:layer.revision};
      },
      drawArtifactExamples: (part, role, tool) => {
        const layer = state.layers.find(l => l.id === (role==='positive'?'artifact-':'artifact-examples-') + part);
        if (!(layer?.editable||layer?.preview?.editable)) return false;
        if(role==='negative')layer.default_class = 'Good tissue';
        layer.visible = true; layer.open = true; state.drawInto = layer.id;
        state.selected = null; state.picks = null; state.selectMode = false;
        setTool(['rect','polygon','freehand'].includes(tool || state.tool) ? (tool || state.tool) : 'polygon');
        renderShapes(); buildPanel(); saveView(); return true;
      },
      artifactExampleState: part => {
        const layer = state.layers.find(l => l.id === 'artifact-examples-' + part);
        const artifact = state.layers.find(l => l.id === 'artifact-' + part);
        const role = f => f.properties.example_role || (f.properties.classification?.name === 'Artifact' ? 'positive' : 'negative');
        return {revision:layer?.revision,
          positive:(artifact?.features || []).length,
          negative:(layer?.features || []).filter(f => role(f) === 'negative').length,
          drawing:state.drawInto===artifact?.id?'Artifact':state.drawInto === layer?.id ? layer?.default_class : null,
          preview:!!artifact?.preview,
          busy:!!(layer?.brushBusy || layer?.loadingShapes), error:layer?.shapesLoadError || layer?.saveError};
      },
      finishArtifactExamples: async part => {
        const layer = state.layers.find(l => l.id === 'artifact-examples-' + part);
        if (!layer || layer.brushBusy || layer.loadingShapes) throw Error('Wait for the examples to finish loading or editing.');
        if ([layer.id,'artifact-'+part].includes(state.drawInto) && drawing.points?.length >= 3) finishPolygon();
        await flushAllShapes();
        await flushShapeSave(layer);
        if (layer.saveError || layer.shapesLoadError) throw Error(layer.saveError || layer.shapesLoadError);
      },
      reloadArtifactExamples: async part => {
        for(const id of ['artifact-'+part,'artifact-examples-'+part]){
          const layer=state.layers.find(l=>l.id===id);
          if(layer&&!layer.preview)await loadShapes(layer);
        }
        renderShapes();buildPanel();
      },
      artifactNeedsReview: part => {
        const layer = state.layers.find(l => l.id === 'artifact-' + part);
        return !!(layer?.preview && layer.history?.length);
      },
      ensureTissueLayer: async spec => {
        if (!spec || !state.layers.some(l => l.type === 'image' && l.token === spec.image_token)) return;
        if (!state.layers.some(l => l.id === 'tissue-regions')) await addLayerToView({...spec});
      },
      showNeckReview,
      approveNeck: async neck => {
        const layer = state.layers.find(l => l.id === (state.tissueLayerId || 'tissue-regions'));
        if(layer)layer.visible=true;
        return layer && applyPolygonEdit(layer, layer.features.filter(f=>!layer.working_class||f.properties.classification?.name===layer.working_class)
          .map(f => f.properties.id), {op:'neck',cutout:neck.geometry});
      },
      tissueForPieces: () => {
        const layer = state.layers.find(l => l.id === (state.tissueLayerId || 'tissue-regions'));
        return layer ? {features:layer.features, preview:!!layer.preview, revision:layer.revision,
          token:layer.token, image_token:layer.image_token,
          busy:!!(layer.brushBusy || layer.preview?.saving || layer.saveBusy || layer.savePending || layer.loadingShapes)} : null;
      },
      applyPiecePreview: (features, original) => {
        const layer = state.layers.find(l => l.id === (state.tissueLayerId || 'tissue-regions'));
        if (!layer?.preview || layer.features !== original || layer.brushBusy || layer.preview.saving) return false;
        pushHistory(layer); state.lastEdited = layer; layer.features = features;
        state.selected = null; state.picks = null;
        saveShapes(layer); renderShapes(); buildPanel(); refreshRegionList(); return true;
      },
      zoomAnnotationBox: bounds => { const item = baseItem(); if (item) state.viewer.viewport.fitBounds(item.imageToViewportRectangle(...bounds)); },
      clearTissuePreview: () => {
        const layer = state.layers.find(l => l.id === (state.tissueLayerId || 'tissue-regions'));
        if (layer) clearShapePreview(layer);
      },
      annotationState: () => Object.fromEntries(shapeLayers().map(l => [l.id, polygonSaveState(l)])),
      reloadAnnotations: async () => {
        const image = state.layers.find(l=>l.type==='image');
        if (image && state.imagesReady) {
          const found = await getJSON('/studio/'+encodeURIComponent(session)+'/annotation_sources?layers=1');
          if (!state.layers.includes(image)) return;
          for (const spec of found.layers || []) {
            if (!state.layers.some(l=>l.id===spec.id || (l.type===spec.type && l.token===spec.token)))
              await addLayerToView({...spec,visible:false});
          }
        }
        for (const layer of shapeLayers().filter(l => l.annotation_kind)) {
          if (layer.preview || layer.savePending || layer.saveBusy || layer.brushBusy) continue;
          const before = layer.revision;
          const result = await shapeRequest(layer, '?revision=' + encodeURIComponent(before || ''));
          if (!state.layers.includes(layer) || layer.preview || layer.savePending || layer.saveBusy || layer.brushBusy || before !== layer.revision || result.unchanged) continue;
          if (result.revision !== before) {
            const previous=JSON.stringify(layer.features||[]);
            clearShapePreview(layer, false);
            layer.features = result.features; layer.revision = result.revision;
            layer.history=[...(layer.history||[]),previous];layer.future=[];layer.historyCache=null;state.lastEdited=layer;
            layer.shapesLoadError = null;
            layer.status = loadedShapesText(layer);
            if (state.selected?.layer === layer) state.selected = null;
            if (state.picks?.layer === layer) state.picks = null;
          }
        }
        renderShapes(); buildPanel();
      },
      // A shapes layer's list and tools (Studio's drawn artifacts, regions), by the layer's id; false when it is not open.
      renderShapeLayer: (holder, id) => { const restoreFocus=keepRegionLabelFocus(holder);const layer = state.layers.find((l) => l.type === 'shapes' && l.id === id); holder.replaceChildren(); if (!layer) return false; (layer.regions ? regionsControls : shapesControls)(layer, holder); restoreFocus();return true; },
      viewBounds: () => {
        const item = baseItem();
        if (!item || !state.viewer) return null;
        const box = item.viewportToImageRectangle(state.viewer.viewport.getBounds(true));
        return [box.x, box.y, box.width, box.height];
      },
      imageSize: () => (state.base && state.base.info ? { width: state.base.info.width, height: state.base.info.height, microns_per_pixel:state.base.info.pixel_size_um } : null),
      imageMarkers: () => {
        const image = state.layers.find((l) => l.type === "image" && l.channels && !(l.info && l.info.rgb));
        return image ? image.channels.map((channel) => channel.name) : [];
      },
      selectedImageMarkers: () => (state.layers.find(l=>l.type==='image'&&l.channels)?.channels||[]).filter(c=>c.visible).map(c=>c.name),
      showClassificationPolygons: async found => {
        let layer=state.layers.find(l=>l.id===found.layer.id);
        if(layer&&layer.token!==found.layer.token){removeLayer(layer);layer=null;}
        if(!layer)layer=await addLayerToView({...found.layer,visible:true});
        await loadShapes(layer);
        if(found.before){layer.history=[...(layer.history||[]),JSON.stringify(found.before)];layer.future=[];state.lastEdited=layer;}
        layer.visible=true;layer.open=true;renderShapes();buildPanel();saveView();
      },
      heCompare,
      showSegPreview: (preview) => { state.segPreview = preview && preview.outlines ? preview : null; showPreview("seg", state.segPreview); },
      showPreview,
      tlsEditor: tlsBoundaryEditor,
      showRaster,
      drawExample,
      saveFigure,
      backgroundLevels,
      renderThresholdBenchmark,
      clearThresholdPreview,
      // Manual reference labelling lives in Studio's Phenotyping step, not in the layer list.
      renderLabels: holder => {
        const cells = state.layers.find(item => item.type === 'cells' && item.combinedTable);
        holder.replaceChildren();
        if (!cells) { holder.append(manualReferenceCard(null)); return; }
        const place = () => {
          if (!holder.isConnected || !state.layers.includes(cells)) return;
          if (cells.table !== cells.combinedTable) { cells.table=cells.combinedTable;cells.columns=null;cells.live=null;bump(cells); }
          holder.replaceChildren(manualReferenceCard(cells));
        };
        if (state.imagesReady) place(); else state.whenImagesReady.then(place);
      },
      referenceColumn: () => state.labelColumn || 'phenotype_ref1',
      cellsState: () => {
        const cells = state.layers.find(item => item.type === 'cells' && (item.tables || []).length);
        return cells ? {color_by: cells.color_by || '', live: !!cells.live, resolve: !!cells.resolve} : null;
      },
      // What the viewer shows now, for the assistant: visible markers and their display range, the cells'
      // colouring and hidden groups, the other layers, and the area on screen (image pixels).
      viewSummary: () => {
        if (!state.spec) return null;
        const image = state.layers.find(l => l.type === 'image' && l.channels);
        const cells = state.layers.find(l => l.type === 'cells');
        const item = baseItem();
        let area = null;
        if (item && state.viewer) { const box = item.viewportToImageRectangle(state.viewer.viewport.getBounds(true)); area = [box.x, box.y, box.width, box.height].map(Math.round); }
        const info = state.base && state.base.info;
        return {
          image: state.spec.title || '',
          image_token: image?.token || '',
          size: info ? [info.width, info.height] : null,
          area_on_screen: area,
          markers_shown: image ? image.channels.filter(c => c.visible).map(c => ({name: c.name, colour: c.color, low: c.low, high: c.high})) : [],
          cells: cells ? {shown: !!cells.visible, mode: cells.mode, coloured_by: cells.color_by || '', marker_transforms:cells.markerTransforms || [], hidden_groups: Array.from(cells.hidden || []).slice(0, 60)} : null,
          layers: state.layers.filter(l => l.visible && l.type !== 'image').map(l => l.name || l.type).slice(0, 30),
        };
      },
      // Studio's heatmaps: colour the cells by a result column and show only the picked groups (all when none).
      // Answers false when no open cell layer has the column.
      // Studio's Cellpose training: cells to correct, kept in this page only (never written to a file).
      // New drawings go to this layer (it is the topmost editable one); Delete removes the selected cell.
      editCells: (name, polygons) => {
        let layer = state.layers.find((l) => l.type === "shapes" && l.local === name);
        const features = polygons.map((ring, index) => {
          const closed = ring.concat([ring[0]]).map((p) => [Number(p[0]), Number(p[1])]);
          return { type: "Feature", geometry: { type: "Polygon", coordinates: [closed] },
            properties: { id: newId(), name: "Cell " + (index + 1), label: "Cell " + (index + 1), classification: { name: "Cell", color: rgbOf("#00e5ff") } } };
        });
        if (!layer) {
          // The image's information (the layer card reads it), not its token: the cells are not Auto regions.
          const image = state.layers.find((l) => l.type === "image");
          layer = { id: "training-" + Date.now().toString(36), type: "shapes", local: name, name, visible: true, opacity: 1,
            editable: true, features: [], history: [], hidden: new Set(), open: true, info: image ? image.info : {},
            classes: [{ name: "Cell", color: "#00e5ff" }] };
          state.layers.push(layer);
        } else pushHistory(layer);
        layer.features = features;
        renderShapes(); updateToolbar(); buildPanel();
        return features.length;
      },
      // The next click on the image, in image pixels (Studio places a training tile there); Esc gives null.
      pickPoint: () => new Promise((resolve) => {
        if (state.pointCallback) state.pointCallback(null);
        state.pointCallback = resolve; setTool("off"); updateToolbar();
        message("Click where the tile goes; Esc cancels.");
      }),
      readCells: (name) => {
        const layer = state.layers.find((l) => l.type === "shapes" && l.local === name);
        return layer ? layer.features.flatMap((f) => polygonsOf(f.geometry).map((rings) => rings[0])) : null;
      },
      dropCells: (name) => {
        state.layers = state.layers.filter((l) => !(l.type === "shapes" && l.local === name));
        if (state.selected && !state.layers.includes(state.selected.layer)) state.selected = null;
        renderShapes(); updateToolbar(); buildPanel();
      },
      // Studio's Segmentation step shows N and M toggles on each marker; null takes them away.
      setMarkerRoles: roles => { state.markerRoles = roles || null; buildPanel(); },
      showCellGroups: async (column, shown, all) => {
        // A command still running (a group's top markers, say) returns the view it was sent with: let it land
        // first, so the groups asked for now are not put back to the earlier ones.
        await (state.commandQueue || Promise.resolve()).catch(() => {});
        await state.phenotypingLink;
        const cells = state.layers.find(item => item.type === 'cells' && item.combinedTable) || (!dataView.combined && state.layers.find(item => item.type === 'cells' && (item.tables || []).length));
        if (!cells || !(await cellTableWith(cells, [column], cells.combinedTable))) {
          document.dispatchEvent(new CustomEvent('spacemap-groups-picked', {detail:{layer:null,column,values:shown || []}}));
          return false;
        }
        cells.live = null; cells.color_by = column;
        cells.hidden = new Set(shown && shown.length ? all.filter(value => !shown.includes(value)) : []);
        if (cells.mode === 'outline') cells.mode = 'both';
        if (!cells.visible) setVisible(cells, true);
        bump(cells);
        document.dispatchEvent(new CustomEvent('spacemap-groups-picked', {detail:{layer:cells,column,values:shown || []}}));
        return true;
      },
      refreshPhenotypingLinks: () => (state.phenotypingLink = dataRefreshCombined()),
      showCombinedData: async () => { await (state.phenotypingLink || dataRefreshCombined()); showDataTab(); await dataChoose('combined'); },
      // Studio's Background step: an image's saved levels, set as its markers' minimums when it opens.
      applyLevels: (levels) => applyBackgroundLevels(levels),
      // Studio's Background step: Auto all markers (not the blank cycles) on the open image, with its Auto method and "in".
      detectLevels: async () => {
        const image = state.layers.find((l) => l.type === "image" && l.inputs && !(l.info && l.info.rgb));
        if (!image) return false;
        await runMarkerAuto(image, image.channels.filter((c) => !/blank|empty|control/i.test(c.name || "")));
        return true;
      },
      // Studio's Background step: Auto measures inside a class of the image's annotation layer (or the whole image).
      setBackgroundScope: (annotationId, className) => {
        const image = state.layers.find((l) => l.type === "image" && !(l.info && l.info.rgb));
        const annotation = state.layers.find((l) => l.type === "annotations" && l.id === annotationId);
        if (!image) return;
        image.autoScope = annotation ? JSON.stringify(className ? { annotation: annotation.token, class: className } : { annotation: annotation.token }) : "";
        remember(image); buildPanel();
      },
      // Cell tagging's preview: colour the ROI's cells by a column computed in memory (with or without a cell table).
      previewCells: column => {
        const cells = state.layers.find(item => item.type === 'cells');
        if (!cells) return false;
        cells.live = null; cells.color_by = column || null; cells.hidden = new Set();
        if (column && cells.mode === 'outline') cells.mode = 'both';
        if (!cells.visible) setVisible(cells, true);
        bump(cells);
        return true;
      },
      colourCells: async column => {
        await state.phenotypingLink;
        const cells = state.layers.find(item => item.type === 'cells' && item.combinedTable) || (!dataView.combined && state.layers.find(item => item.type === 'cells' && (item.tables || []).length));
        if (!cells) return;
        if (column && !(await cellTableWith(cells,[column],cells.combinedTable))) { message('This cell layer has no linked ' + column + ' values.',true); return; }
        cells.live = null; cells.color_by = column; cells.hidden = new Set();
        if (column && cells.mode === 'outline') cells.mode = 'both';
        if (!cells.visible) setVisible(cells, true);
        bump(cells);
      },
      openImage: image => runCommand('open_image', {image:image.group + ' | ' + image.name})});
  }
  setUpImagePicker(session);
  const poll = async () => {
    try {
      const known = state.sessionVersion;
      const result = await getJSON("/session/" + encodeURIComponent(session) + (known !== null ? "?v=" + known : ""));
      if (!state.commandBusy && result.spec && result.version > (state.sessionVersion ?? -1)) {
        document.title = "SPACEMAP viewer: " + (result.spec.title || "");
        if (result.spec.command) await acceptCommandView(result);
        else { await init(result.spec); state.sessionVersion = result.version; }
      }
      state.unreachable = false;
    } catch (error) {
      // SPACEMAP restarting, or not yet reopened: keep trying quietly.
      if (!state.unreachable) message("Waiting for SPACEMAP (" + error.message + "); this window reconnects by itself.", true);
      state.unreachable = true;
    }
    setTimeout(poll, state.unreachable ? 4000 : 2000);
  };
  if (!session) { message("Open this window from SPACEMAP's image viewer.", true); return; }
  poll();
}

function commandView() {
  if (!state.spec) return undefined;
  const view = {id:state.spec.id, layers:state.layers.map(layer => ({
    id:layer.id, visible:layer.visible, opacity:layer.opacity, channels:layer.channels,
    table:layer.table, color_by:layer.color_by, mode:layer.mode, hidden:Array.from(layer.hidden || [])
  }))};
  const item = baseItem();
  if (item && state.viewer) {
    const bounds = item.viewportToImageRectangle(state.viewer.viewport.getBounds(true));
    view.bounds = [bounds.x, bounds.y, bounds.width, bounds.height];
  }
  return view;
}

function captureView() {
  const source = state.viewer?.drawer?.canvas;
  if (!source || !state.imagesReady) throw new Error('Wait for the image to appear first.');
  const canvas = document.createElement('canvas'), scale = Math.min(1, 1024 / Math.max(source.width, source.height));
  canvas.width = Math.max(1,Math.round(source.width * scale)); canvas.height = Math.max(1,Math.round(source.height * scale));
  const context = canvas.getContext('2d'); context.fillStyle = '#000'; context.fillRect(0,0,canvas.width,canvas.height);
  context.drawImage(source,0,0,canvas.width,canvas.height);
  const item = baseItem(), ratio = canvas.width / Math.max(1,source.clientWidth || source.width);
  if (item) drawShapesOnCanvas(context, (x,y) => {const point = item.imageToViewerElementCoordinates(new OpenSeadragon.Point(x,y));return [point.x*ratio,point.y*ratio];},ratio);
  return canvas.toDataURL('image/png');
}

function runCommand(name, args = {}, options = {}) {
  const task = (state.commandQueue || Promise.resolve()).catch(() => {}).then(() => executeViewerCommand(name,args,options));
  state.commandQueue = task;
  return task;
}

async function executeViewerCommand(name, args = {}, options = {}) {
  if (state.commandBusy) throw new Error('Wait for the current action to finish.');
  state.commandBusy = true;
  try {
  const session = state.spec?.session || new URLSearchParams(location.search).get('session');
  await flushLabels();
  if (labelSaves.queue.length || labelSaves.busy) throw new Error('Save waiting labels before running this action.');
  const body = {name, args, request_id:newSaveId(), version:state.sessionVersion, view:commandView(), ...options};
  const result = await postJSON('/studio/' + encodeURIComponent(session) + '/command', body);
  if (result.status === 'done') await acceptCommandView(result, true);
  return result;
  } finally { state.commandBusy = false; }
}

async function applyCommandChannels(layer, channels) {
  const before = JSON.stringify(layer.channels);
  const byIndex = new Map(channels.map(channel => [channel.index, channel]));
  layer.channels = layer.channels.map(channel => {
    const next = {...channel, ...byIndex.get(channel.index)};
    for (const key of ['low', 'high']) if (!Number.isFinite(next[key])) next[key] = undefined;
    return next;
  });
  await fillLimits(layer, layer.channels.filter(channel => channel.visible && (channel.low === undefined || channel.high === undefined)));
  if (JSON.stringify(layer.channels) !== before) { remember(layer); refreshLayer(layer); }
}

function acceptCommandView(result, initiated = false) {
  const task = (state.viewQueue || Promise.resolve()).catch(() => {}).then(() => applyCommandView(result, initiated));
  state.viewQueue = task;
  return task;
}

async function applyCommandView(result, initiated = false) {
  if (result.version < (state.sessionVersion ?? -1)) return;
  const spec = result.spec;
  if (!spec) return;
  const event = spec.command || {id:result.command_id, effects:result.effects || []};
  if (state.lastCommand === event.id && state.sessionVersion === result.version) return;
  const firstSessionView = !state.spec && !initiated;
  // Same-image actions keep the existing viewer and its decoded image tiles.
  const reopened = !state.spec || state.spec.id !== spec.id;
  const effects = result.effects || event.effects || [];
  const commandName = event.name || result.name;
  const navigation = ['open_image','open_annotation_image','open_roi_image','open_background_image'].includes(commandName) ||
    (reopened && !commandName && !effects.length);  // saved opening commands from older Studio versions
  if (reopened) await init(spec);
  if (state.spec?.id !== spec.id) return;
  // init has already restored this image's preferences. Opening is not a
  // display edit: never replay the catalog's presets over those preferences.
  for (const source of reopened && (navigation || firstSessionView) ? [] : uniqueViewLayers(spec.layers || [])) {
    if (navigation && viewRemoved(source)) continue;
    let layer = state.layers.find(item => item.id === source.id);
    const kept = {};
    if (navigation && layer) VIEW_FIELDS.forEach(key => { if (layer[key] !== undefined) kept[key] = layer[key]; });
    // The same id for another image (the H&E to align after a registered image's H&E, say): replace the layer.
    if (layer && source.token && layer.token !== source.token) { removeLayer(layer); layer = null; }
    if (!layer) {
      await addLayerToView({...source, ...(navigation ? (source.added ? kept : {visible:false, ...kept}) : {})});  // a step's own layers show
      continue;
    }
    // Upgrade remembered copies from older Studio versions in place.
    for (const key of ['regions','annotation_kind','classes','default_class','names','saved_file']) {
      if (source[key] !== undefined) layer[key] = source[key];
    }
    if (navigation) continue;
    if (source.type === 'image' && source.channels) await applyCommandChannels(layer, source.channels);
    let redraw = false;
    for (const field of VIEW_FIELDS.filter(key => !['visible','opacity','open'].includes(key))) {
      // These controls are shared by dataset in the browser. Unrelated job
      // results can carry an old per-image consensus default in their spec.
      if(layer.type==='cells'&&layer.referenceChoiceTable&&
          ['suggestionsOn','suggestionMode','suggestionModelId','learnColumn'].includes(field)&&
          !effects.some(effect=>effect.type==='restore_view'))continue;
      // Combined linking replaces aliases of the same table. A delayed
      // command must not put one of those removed aliases back in the picker.
      if(field==='table'&&layer.table===layer.combinedTable&&
          !(layer.tables || []).some(table=>table.id===source.table))continue;
      if (source[field] !== undefined && JSON.stringify(layer[field]) !== JSON.stringify(source[field])) {
        layer[field] = source[field]; redraw = true;
      }
    }
    if (source.hidden && JSON.stringify(Array.from(layer.hidden || [])) !== JSON.stringify(source.hidden)) {
      layer.hidden = new Set(source.hidden); redraw = true;
    }
    if (source.visible !== undefined && layer.visible !== source.visible) setVisible(layer, source.visible);
    if (source.opacity !== undefined && layer.opacity !== source.opacity) setOpacity(layer, source.opacity);
    if (redraw) refreshLayer(layer);
  }
  for (const effect of effects) {
    const layer = state.layers.find(item => item.id === effect.layer);
    if (effect.type === 'restore_view') {
      state.layers.filter(item => item.added && !spec.layers.some(source => source.id === item.id)).forEach(removeLayer);
    }
    if (['cells','labels'].includes(effect.type) && layer) {
      layer.live = null; layer.columns = null; state.versions[layer.id] = (state.versions[layer.id] || 0) + 1; refreshLayer(layer);
    }
    if (effect.type === 'threshold' || effect.type === 'refresh_resolve') {
      for (const target of state.layers.filter(item => item.resolve && (item === layer || effect.type === 'refresh_resolve'))) {
        target.live = await postJSON('/resolve/' + target.resolve + '/open', {});
        target.liveHidden = target.liveHidden || new Set(); syncResolveRanges(target); refreshLayer(target);
      }
    }
    if (effect.type === 'refresh_labels') {
      state.layers.filter(item => item.type === 'cells').forEach(item => { item.columns = null; labelSaves.changed.add(item); });
      refreshLabelColours();
    }
    if (effect.type === 'reload_shapes' && layer && layer.type === 'shapes') { await loadShapes(layer); refreshLayer(layer); }  // its file was rewritten
    if (effect.type === 'zoom_cell' && layer) inspectReviewedCell(layer, effect.cell);
    if (effect.type === 'save_image' && initiated && !result.replayed) saveImage();
  }
  if (!firstSessionView && spec.bounds && effects.some(effect => ['restore_view','zoom_cell'].includes(effect.type))) {
    const applyBounds = () => { const item = baseItem(); if (item) state.viewer.viewport.fitBounds(item.imageToViewportRectangle(...spec.bounds), true); };
    if (baseItem()) applyBounds(); else state.whenImagesReady?.then(applyBounds);
  }
  state.spec = spec; state.sessionVersion = result.version; state.lastCommand = event.id;
  buildPanel(); saveView(); state.studio?.viewChanged(spec);
}

// The window's own image list: any of the project's images opens here.
function setUpImagePicker(session) {
  const picker = $("image-picker");
  picker.hidden = false;
  picker.innerHTML = "";
  picker.appendChild(el("option", { value: "", text: "Open another image..." }));
  let loaded = false;
  const fill = (images) => {
    picker.innerHTML = "";
    picker.appendChild(el("option", { value: "", text: "Open another image..." }));
    let group = null, holder = picker;
    (images || []).forEach((image) => {
      if (image.group !== group) { group = image.group; holder = el("optgroup", { label: group }); picker.appendChild(holder); }
      holder.appendChild(el("option", { value: String(image.id), text: image.name }));
    });
    picker.appendChild(el("option", { value: "add", text: "Add an image from outside the project..." }));
  };
  const load = async () => {
    if (loaded) return;
    loaded = true;
    try {
      fill((await getJSON("/session/" + encodeURIComponent(session) + "/images")).images);
    } catch (error) { loaded = false; message(error.message, true); }
  };
  picker.addEventListener("focus", load);
  picker.addEventListener("mousedown", load);
  picker.addEventListener("change", async () => {
    if (picker.value === "") return;
    if (picker.value === "add") {
      picker.value = "";
      const path = window.prompt("Full path of the image to add (QPTIFF, OME-TIFF, TIFF or SVS). " +
        "It is added to this project's list; the file is not copied or changed.");
      if (!path || !path.trim()) return;
      try {
        const result = await postJSON("/session/" + encodeURIComponent(session) + "/add-image", { path: path.trim() });
        fill(result.images);
        loaded = true;
        picker.value = String(result.added);
        picker.dispatchEvent(new Event("change"));
      } catch (error) { message("Could not add the image: " + error.message, true); }
      return;
    }
    const choice = picker.options[picker.selectedIndex].text;
    message("Opening " + choice + "...");
    try {
      const result = await postJSON("/session/" + encodeURIComponent(session) + "/open", { id: Number(picker.value) });
      state.sessionVersion = result.version;
      document.title = "SPACEMAP viewer: " + (result.spec.title || "");
      await init(result.spec);
    } catch (error) { message("Could not open " + choice + ": " + error.message, true); }
    picker.value = "";
  });
}

// ---------------------------------------------------------------- helpers
function api(path) {
  const spec = state.spec || {};
  let root = state.apiBase || "";
  if (STANDALONE && state.apiBase) root = state.apiBase;
  else if (spec.api) root = spec.api_port ? (location.protocol + "//" + location.hostname + ":" + spec.api_port + spec.api) : spec.api;
  return root.replace(/\/$/, "") + path;
}
async function getJSON(path, options = {}) {
  const response = await fetch(api(path), { credentials: "same-origin", signal: options.signal });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || ("HTTP " + response.status));
  if(body.workflow_warning)message(body.workflow_warning,true);
  return body;
}
async function postJSON(path, payload, options = {}) {
  const response = await fetch(api(path), {
    method: "POST", credentials: "same-origin", signal: options.signal,
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || ("HTTP " + response.status));
  if(body.workflow_warning)message(body.workflow_warning,true);
  return body;
}
// A POST whose answer arrives as JSON lines while it is made: each line goes to onEvent, the last ({done}) is returned.
async function postStream(path, payload, onEvent, {signal} = {}) {
  const response = await fetch(api(path), {
    method: "POST", credentials: "same-origin",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal,
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || ("HTTP " + response.status));
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let at;
    while ((at = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, at).trim();
      buffer = buffer.slice(at + 1);
      if (!line) continue;
      const event = JSON.parse(line);
      if (event.error) throw new Error(event.error);
      if (event.done) return event.done;
      onEvent(event);
    }
    if (done) throw new Error("The answer stopped before it was finished.");
  }
}
function el(tag, attrs, children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null) node.setAttribute(key, value);
  });
  (children || []).forEach((child) => child && node.appendChild(child));
  return node;
}
let messageTimer = null;
// ---------------------------------------------------------------- building blocks
// Panels show short labels and controls; explanations sit behind a (?) that
// shows on hover or keyboard focus, as in classic SPACEMAP.
// The (?) help text floats above every panel (a panel that scrolls would clip it), beside the (?) and kept inside
// the window: below it when there is room, else above; never past the left or right edge.
(() => {
  let tip = null, owner = null;
  const place = target => {
    if (!tip) { tip = document.createElement('div'); tip.id = 'floating-tip'; tip.setAttribute('role', 'tooltip'); }
    (target.closest('dialog')||document.body).appendChild(tip);
    tip.textContent = target.dataset.tip || ''; tip.hidden = !tip.textContent; owner = target;
    if (tip.hidden) return;
    const box = target.getBoundingClientRect(), width = Math.min(280, window.innerWidth - 16);
    tip.style.width = width + 'px';
    const left = Math.max(8, Math.min(box.right - width, window.innerWidth - width - 8));
    tip.style.left = left + 'px'; tip.style.top = '0px';
    const height = tip.offsetHeight;
    const below = box.bottom + 6, above = box.top - 6 - height;
    tip.style.top = (below + height <= window.innerHeight - 8 || above < 8 ? Math.min(below, window.innerHeight - height - 8) : above) + 'px';
  };
  const hide = target => { if (tip && (!target || target === owner)) { tip.hidden = true; owner = null; } };
  document.addEventListener('mouseover', event => { const target = event.target.closest?.('.help[data-tip]'); if (target) place(target); });
  document.addEventListener('mouseout', event => { const target = event.target.closest?.('.help[data-tip]'); if (target && !target.contains(event.relatedTarget)) hide(target); });
  document.addEventListener('focusin', event => { const target = event.target.closest?.('.help[data-tip]'); if (target) place(target); });
  document.addEventListener('focusout', event => { const target = event.target.closest?.('.help[data-tip]'); if (target) hide(target); });
  document.addEventListener('scroll', () => hide(), true);
})();

function help(text) {
  return el("span", { class: "help", tabindex: "0", role: "note", "aria-label": text, "data-tip": text, text: "?" });
}
function chip(text, kind, title) {
  return el("span", { class: "chip" + (kind ? " " + kind : ""), text: text, title: title || null });
}
function field(label, control, tip) {
  return el("div", { class: "row field" }, [el("label", { class: "lbl", text: label }), control, tip ? help(tip) : null]);
}

function message(text, isError) {
  const box = $("message");
  box.textContent = text;
  box.className = isError ? "error" : "";
  box.hidden = false;
  clearTimeout(messageTimer);
  messageTimer = setTimeout(() => { box.hidden = true; }, isError ? 7000 : 3500);
}
const debouncers = {};
function debounce(key, fn, wait) {
  clearTimeout(debouncers[key]);
  debouncers[key] = setTimeout(fn, wait || 250);
}
function hex(color) { return String(color || "#ffffff").replace("#", "").slice(0, 6); }
function storageKey(layer) { return "spacemap-viewer:" + (layer.display_token || layer.token) + ':' + layer.id; }
function channelSettings(layer) {
  return layer.channels.map(c => ({index:c.index, name:c.name, label:c.label || c.name,
    color:c.color, low:c.low, high:c.high, visible:c.visible, custom:!!c.custom}));
}
function remember(layer) {
  if (!state.layers.includes(layer)) return;  // an earlier image finished measuring
  if (layer.display_token) {  // saved with the project, where other steps can read it
    queueViewerSetting('/display/' + layer.display_token + '/save', {channels:channelSettings(layer),
      auto:{method:layer.autoMethod || 'Triangle', scope:layer.autoScope || '', parameters:layer.autoParameters, native:!!layer.autoNative, display:layer.displayAuto, histogram:layer.histogramView}});
  }
  try { localStorage.setItem(storageKey(layer) + ":auto", JSON.stringify({ method: layer.autoMethod, scope: layer.autoScope, parameters:layer.autoParameters, native:!!layer.autoNative, display:layer.displayAuto, histogram:layer.histogramView })); } catch (e) { /* private mode */ }
  try { localStorage.setItem(storageKey(layer), JSON.stringify(layer.channels.map((c) => [c.visible, c.color, c.low, c.high, !!c.custom]))); } catch (e) { /* private mode */ }
  saveView();
}
function recall(layer) {
  try { return JSON.parse(localStorage.getItem(storageKey(layer)) || "null"); } catch (e) { return null; }
}

// The first of a cell layer's tables (its current one first) that has every column in *names*; the layer switches to it.
async function cellTableWith(cells, names, onlyTable) {
  const ids = (onlyTable ? [onlyTable] : [cells.table, ...(cells.tables || []).map(t => t.id)]).filter((id, i, all) => id && all.indexOf(id) === i);
  for (const id of ids) {
    try {
      const found = await getJSON("/cells/" + cells.token + "/columns?t=" + encodeURIComponent(id));
      if (names.every(name => (found.columns || []).includes(name))) {
        if (cells.table !== id) { cells.table = id; cells.columns = null; }
        return id;
      }
    } catch (_) { /* the next table */ }
  }
  return null;
}
