/* SPACEMAP viewer, part 7 of 12: layer order, added layers and density maps.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- layer order and added layers
function consolidateCellResults() {
  // Older Show result commands cloned the segmentation. Keep its original
  // layer; otherwise a second overlay makes cluster filtering look broken.
  for (const copy of [...state.layers]) {
    if (copy.type !== 'cells' || !copy.id.startsWith('result-')) continue;
    const original = state.layers.find(layer => layer.type === 'cells' && !layer.id.startsWith('result-') &&
      layer.token === copy.token && layer.image_token === copy.image_token);
    if (original) removeLayer(copy);
  }
}

function moveLayer(layer, step) {
  const tiles = state.layers.filter((l) => l.type !== "shapes");
  const at = tiles.indexOf(layer), to = at + step;
  if (to < 0 || to >= tiles.length) return;
  const other = tiles[to];
  const i = state.layers.indexOf(layer), j = state.layers.indexOf(other);
  state.layers[i] = other; state.layers[j] = layer;
  restack();
  buildPanel();
  saveView();
}

// gone: the layer was deleted from the project (not only from this view), so the view does not remember hiding it.
function removeLayer(layer, gone = false) {
  removeAnnotationOutline(layer);state.annotationSelections?.delete(layer);
  if (gone) state.removedLayers?.delete(viewLayerKey(layer));
  else if (state.spec?.layers?.some(source => source.id === layer.id)) state.removedLayers?.add(viewLayerKey(layer));
  retire(layer);
  if (layer.item) { state.viewer.world.removeItem(layer.item); layer.item = null; }
  state.layers = state.layers.filter((l) => l !== layer);
  if (state.selected?.layer === layer) state.selected = null;
  if (state.picks?.layer === layer) state.picks = null;
  if (layer.type === 'shapes') { renderShapes(); updateToolbar(); refreshRegionList(); annotationsChanged(layer); }
  buildPanel();
  saveView();
}

async function addLayerToView(layer, below) {
  const generation = state.initGeneration;
  const pending = state.layerAdds || (state.layerAdds = new Map());
  const task = (pending.get(layer.id) || Promise.resolve()).catch(() => {}).then(() => {
    if (generation === state.initGeneration) return insertLayerToView(layer, below, generation);
  });
  pending.set(layer.id, task);
  try { return await task; }
  finally { if (pending.get(layer.id) === task) pending.delete(layer.id); }
}

async function insertLayerToView(layer, below, generation) {
  if (layer.type === 'annotations' && layer.saved_file) {
    const working = state.layers.find(other => other.type === 'shapes' && other.saved_file === layer.saved_file);
    if (working) return working;
  }
  for (const old of state.layers.filter(item => item.id === layer.id && item.type === 'shapes')) await flushShapeSave(old);
  if (generation !== state.initGeneration) return;
  const token = layer.type === "image" ? layer.token : layer.image_token;  // an image layer is its own image
  if (!state.infos[token]) state.infos[token] = await getJSON("/image/" + token + "/info");
  layer.info = state.infos[token];
  if (layer.type === "image" && !layer.info.rgb) await prepareChannels(layer);
  layer.visible = layer.visible !== false;
  layer.opacity = layer.opacity === undefined ? 0.8 : layer.opacity;
  layer.opacity = displayOpacity(layer);
  layer.hidden = new Set(layer.hidden || []);
  layer.added = true;
  layer.open = layer.open ?? !layer.annotation_kind;
  let index = state.layers.length;
  if (below) index = Math.max(0, state.layers.indexOf(below));
  // Shapes are drawn on the overlay, not as image tiles: read the saved file first.
  if (layer.type === "shapes") await loadShapes(layer);
  if (generation !== state.initGeneration) return;
  // Replacing a layer keeps a single entry; requests for the same ID are queued.
  const existing = state.layers.filter(item => item.id === layer.id);
  if (layer.type === 'shapes' && layer.saved_file) existing.push(...state.layers.filter(item =>
    item.type === 'annotations' && item.saved_file === layer.saved_file && !existing.includes(item)));
  for (const old of existing) {
    removeLayer(old);
  }
  index = Math.min(index, state.layers.length);
  state.layers.splice(index, 0, layer);
  // Reviewer image rows initially load the image. Its annotation source arrives
  // when an item is opened, after init's normal display-transfer pass.
  if (state.reviewer && layer.type === 'annotations') {
    const matches = (state.displayTransfer?.layers || []).filter(old => old.family === displayFamily(layer));
    if (matches.length === 1) {
      applyDisplayLayout(layer, matches[0].layout);
      applyAnnotationDisplayTransfer(layer, matches[0]);
    }
  }
  state.removedLayers?.delete(viewLayerKey(layer));
  if (layer.type === "shapes") { renderShapes(); updateToolbar(); }
  else if (layer.visible) addLayer(layer);
  buildPanel();
  saveView();
  return layer;
}

function addCard() {
  const card = el("div", { class: "layer" });
  card.appendChild(el("div", { class: "layer-head" }, [el("span", { class: "name", text: "Add to the view" })]));
  const body = el("div", { class: "layer-body" });
  const cells = state.layers.find((l) => l.type === "cells" && (l.tables || []).length);
  const row = el("div", { class: "row wrap" });
  if (cells && !state.studio) {
    row.appendChild(el("button", { class: "btn small", text: "Density map", title: "How crowded the cells are (all cells or chosen phenotypes), or the local average of a column such as a marker",
      onclick: () => {
        const count = state.layers.filter((l) => l.type === "density").length + 1;
        addLayerToView({ id: "density" + count + "-" + Date.now().toString(36), type: "density", name: "Density map " + count,
          token: cells.token, image_token: cells.image_token, cellsId: cells.id, dataMode: "viewer", cellScope: "all", tables: cells.tables, table: cells.table,
          fcol: "", fvals: [], vcol: "", radius: 50, weighted: false, cmap: "#ff3b3b", cmapAuto: true, opacity: 0.75 }, cells);
      } }));
  }
  if (state.spec?.workspace) {
    const file = el("input", { type: "file", accept: ".geojson,.json", hidden: "hidden" });
    file.addEventListener("change", async () => {
      const chosen = file.files && file.files[0];
      if (!chosen) return;
      message("Adding " + chosen.name + "...");
      try {
        const result = await postJSON("/workspace/" + state.spec.workspace + "/add", { name: chosen.name, geojson: await chosen.text() });
        await addLayerToView(result.layer);
        message("Added " + chosen.name + " (" + result.shapes.toLocaleString() + " shapes). It is kept in outputs/viewer/imported.");
      } catch (error) { message("Could not add the file: " + error.message, true); }
      file.value = "";
    });
    row.append(file, el("button", { class: "btn small", text: "GeoJSON file...", title: "Show a GeoJSON file from your computer (QuPath or other tools), in this image's pixel coordinates",
      onclick: () => file.click() }));
    const pick = el("select", { title: "Show one of the project's GeoJSON files" });
    pick.appendChild(el("option", { value: "", text: "Project GeoJSON..." }));
    pick.addEventListener("focus", async () => {
      if (pick.dataset.loaded) return;
      pick.dataset.loaded = "1";
      try {
        const result = await getJSON("/workspace/" + state.spec.workspace + "/files");
        (result.files || []).forEach((f) => pick.appendChild(el("option", { value: f, text: f })));
      } catch (error) { message(error.message, true); }
    }, { once: false });
    pick.addEventListener("change", async () => {
      if (!pick.value) return;
      try {
        const result = await postJSON("/workspace/" + state.spec.workspace + "/add", { relative: pick.value });
        await addLayerToView(result.layer);
      } catch (error) { message("Could not add the file: " + error.message, true); }
      pick.value = "";
    });
    row.appendChild(pick);
  }
  body.appendChild(row);
  if (state.studio) body.appendChild(bringPathRow());  // a file or folder by path; files can also be dropped on the image
  card.appendChild(body);
  return card;
}

// ---------------------------------------------------------------- density maps (as in Tab 9)
function densityControls(layer, body) {
  const owner=resolveDensityData(layer),tables=layer.tables || [];
  if(!layer.table || layer.dataMissing){body.append(el('div',{class:'row'},[
    el('span',{class:'hint grow',text:'No cell data'}),
    el('button',{class:'btn small',text:owner?.table?'Use current cell data':'Choose cell data',onclick:()=>{
      if(owner?.table)setDensityScope(layer,'viewer');
      else document.querySelector('select[aria-label="Cell data"]')?.focus();
    }}),
    help('Choose Cell data at the top of Cells, or import cell data below.')]));return;}
  const grid=el('div',{class:'density-control-grid'}),more=el('details',{class:'density-more'});
  const blocks=state.annotationBlocksOpen||(state.annotationBlocksOpen={});more.open=!!blocks['density-more::'+layer.id];
  more.ontoggle=()=>{if(more.isConnected){blocks['density-more::'+layer.id]=more.open;saveView();}};
  more.append(el('summary',{text:'More'}));body.append(grid);
  const scope=select([['all',layer.dataMode==='fixed'?'All points in this image':'All cells in this image'],
    ...(owner&&layer.dataMode!=='fixed'?[['viewer','Follow shown cells'],['snapshot','Keep this selection']]:[])],layer.cellScope||'all',v=>setDensityScope(layer,v));
  scope.setAttribute('aria-label','Density cells included');
  more.append(cellControlRow('Cells included',[scope],
    'Follow shown cells updates the map with the viewer selection, independently of cell visibility. Keep this selection preserves this table, filters and selected groups while you explore another set. All cells ignores viewer filters. Uses '+cellDataInfo(layer).name+'.'));
  const cellsRow=cellControlRow('Filter by',[]),showRow=cellControlRow('Map value',[]);
  more.append(cellsRow);
  const values = el("div", { class: "legend" });
  more.appendChild(values);
  grid.append(showRow);
  columnsFor(layer).then((columns) => {
    const sorted = sortColumns(columns);
    if(!body.isConnected)return;
    const filter=select([["", "No additional group filter"]].concat(sorted.map((c) => [c, c])), layer.fcol || "",
      (v) => { layer.fcol = v; layer.fvals = []; bump(layer); });filter.setAttribute("aria-label","Density group column");cellsRow.querySelector(".inline").append(filter);
    const metric=select([["", "Cell density (cells/mm²)"]].concat(sorted.filter(c=>layer.columnKinds?.[c]!=="group").map((c) => [c, "Mean " + c])), layer.vcol || "",
      (v) => { layer.vcol = v; bump(layer); });metric.setAttribute("aria-label","Density map value");showRow.querySelector(".inline").append(metric);
  }).catch((error) => cellsRow.appendChild(el("span", { class: "hint", text: error.message })));
  if (layer.fcol) {
    getJSON("/cells/" + layer.token + "/density/groups?t=" + encodeURIComponent(layer.table) + "&col=" + encodeURIComponent(layer.fcol))
      .then((legend) => {
        if (legend.kind !== "categorical") { values.appendChild(el("div", { class: "hint", text: "Choose a column with groups, such as a phenotype." })); return; }
        const first = (legend.items || []).find(item => item.value === layer.fvals?.[0]);
        if (body.isConnected && layer.cmapAuto && first && layer.cmap !== first.color) {
          layer.cmap = first.color; bump(layer); return;
        }
        values.appendChild(el("div", { class: "hint", text: "Tick the groups to count:" }));
        const every = allBox("All groups", (on) => {
          layer.fvals = on ? (legend.items || []).map((i) => i.value) : [];
          layer.name = layer.fvals.length ? "Density: " + (on ? "all groups" : "") : "Density map";
          bump(layer);
        });
        every.sync((legend.items || []).map((i) => (layer.fvals || []).includes(i.value)));
        values.appendChild(every.row);
        (legend.items || []).forEach((item) => {
          const on = (layer.fvals || []).includes(item.value);
          const row = el("div", { class: "item" + (on ? "" : " hidden") }, [
            el("span", { class: "swatch", style: "background:" + item.color }), el("span", { text: item.value }),
            el("span", { class: "count", text: Number(item.count).toLocaleString() })]);
          row.addEventListener("click", () => {
            const chosen = new Set(layer.fvals || []);
            if (chosen.has(item.value)) chosen.delete(item.value); else chosen.add(item.value);
            layer.fvals = Array.from(chosen);
            const names = layer.fvals.slice(0, 3).join(", ") + (layer.fvals.length > 3 ? "..." : "");
            layer.name = layer.fvals.length ? "Density: " + names : "Density map";
            // Until a colour is chosen by hand, a map takes the colour of its first group.
            const first = (legend.items || []).find((i) => i.value === layer.fvals[0]);
            if (layer.cmapAuto && first) layer.cmap = first.color;
            bump(layer);
          });
          values.appendChild(row);
        });
        if ((layer.fvals || []).length > 1) {
          values.appendChild(el("button", { class: "btn small", text: "One map per group",
            title: "Replace this map with one map for each ticked group, each in its group's colour, to compare where they are and where they overlap",
            onclick: async () => {
              for (const value of layer.fvals) {
                const item = (legend.items || []).find((i) => i.value === value) || {};
                await addLayerToView({ id: "density-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 5), type: "density",
                  name: "Density: " + value, token: layer.token, image_token: layer.image_token, tables: layer.tables, table: layer.table, cellsId:layer.cellsId, dataMode:layer.dataMode, cellScope:layer.cellScope, cellDensity:layer.cellDensity, densitySnapshot:layer.densitySnapshot,
                  fcol: layer.fcol, fvals: [value], vcol: layer.vcol, radius: layer.radius, weighted: layer.weighted,
                  cmap: item.color || "#ff3b3b", cmapAuto: true, opacity: layer.opacity }, layer);
              }
              removeLayer(layer);
            } }));
        }
      }).catch(() => {});
  }
  if (layer.vcol) {
    const weighted = el("input", { type: "checkbox" });
    weighted.checked = !!layer.weighted;
    weighted.addEventListener("change", () => { layer.weighted = weighted.checked; bump(layer); });
    more.appendChild(cellControlRow("Fade sparse areas",[weighted]));
  }
  const radius = el("input", { type: "range", min: "5", max: "500", step: "5", value: String(layer.radius || 50) });
  const radiusText = el("span", { class: "hint", text: (layer.radius || 50) + " µm" });
  radius.addEventListener("input", () => { radiusText.textContent = radius.value + " µm"; });
  radius.addEventListener("change", () => { layer.radius = Number(radius.value); bump(layer); });
  const radiusRow=cellControlRow("Radius",[radius,radiusText]);
  const single = String(layer.cmap || "").startsWith("#");
  const colourRow = cellControlRow('Colours',[
    select([["single", "One colour"], ["inferno", "Inferno"], ["magma", "Magma"], ["viridis", "Viridis"], ["plasma", "Plasma"],
      ["turbo", "Turbo"], ["hot", "Hot"], ["Reds", "Reds"], ["Blues", "Blues"]],
      single ? "single" : (layer.cmap || "inferno"),
      (v) => { layer.cmapAuto = false; layer.cmap = v === "single" ? "#ff3b3b" : v; bump(layer); })]);
  if (single) {
    const picker = el("input", { type: "color", value: layer.cmap, title: "The map's colour; the more cells, the stronger" });
    picker.addEventListener("change", () => { layer.cmapAuto = false; layer.cmap = picker.value; bump(layer); });
    colourRow.querySelector('.inline').appendChild(picker);
  }
  grid.appendChild(colourRow);
  const opacity=el('input',{type:'range',min:'0',max:'100',value:String(Math.round((layer.opacity??.75)*100)),'aria-label':'Density opacity'});
  const opacityText=el('span',{class:'hint',text:opacity.value+'%'});
  opacity.oninput=()=>{opacityText.textContent=opacity.value+'%';setOpacity(layer,opacity.value/100);};
  grid.append(cellControlRow('Opacity',[opacity,opacityText]),radiusRow);
  radius.setAttribute('aria-label','Density radius');
  const legend = el("div", { class: "legend", text: "Computing..." });
  body.appendChild(legend);
  body.append(more);
  getJSON("/cells/" + layer.token + "/density/legend?" + densityQuery(layer)).then((result) => {
    if(!legend.isConnected)return;legend.textContent = "";
    const stops = String(result.colormap).startsWith("#") ? "rgba(0,0,0,0)," + result.colormap : gradientStops(result.colormap);
    legend.appendChild(el("div", { class: "gradient", style: "background: linear-gradient(90deg," + stops + ")" }));
    legend.appendChild(el("div", { class: "range" }, [el("span", { text: number(result.min) }), el("span", { text: number(result.max) + " " + result.units })]));
    legend.appendChild(el("div", { class: "hint", text: result.cells.toLocaleString() + " cells, smoothed over " + (layer.radius || 50) + " µm." }));
  }).catch((error) => { legend.textContent = error.message; });
}
