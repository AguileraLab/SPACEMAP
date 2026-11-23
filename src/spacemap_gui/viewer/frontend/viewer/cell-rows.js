// Studio's Cells and maps in Layers: one block per cell layer (Studio's segmentation, an imported one, Voronoi cells or
// circles from a table), each with one short form: Data (only tables holding these very cells), Show, Colour by
// (groups or values) and Filter (which cells are drawn); then the density maps as rows, and Import cells.

function cellsPreviewName(column) {
  return String(column).startsWith("preview:") ? column.slice(8) + " (preview)" : column;
}

function cellRowsOpen() { return state.cellRowsOpen || (state.cellRowsOpen = {}); }

function cellRow({ key, name, kind, shown, onShow, swatch, onRemove, removeTitle, body, data }) {
  const rows = cellRowsOpen();
  const open = !!rows[key] && !!body;
  const card = el("div", { class: "layer annotation-name cell-row" + (open ? "" : " collapsed"), ...data });
  const show = el("input", { type: "checkbox", "aria-label": "Show " + name, title: "Show or hide" });
  show.checked = shown;
  show.addEventListener("click", (event) => event.stopPropagation());
  show.addEventListener("change", () => onShow(show.checked));
  const fold = el("span", { class: "layer-fold", text: body ? (open ? "▾" : "▸") : "", "aria-hidden": "true" });
  const remove = onRemove ? el("button", { class: "btn tiny", text: "×", title: removeTitle, "aria-label": "Remove " + name,
    onclick: (event) => { event.stopPropagation(); onRemove(); } }) : null;
  const head = el("div", { class: "layer-head", onclick: () => { if (!body) return; rows[key] = !open; buildPanel(); } },
    [show, fold, swatch, el("span", { class: "name", text: name, title: name }), kind ? el("span", { class: "kind", text: kind }) : null, remove]
      .filter(Boolean));
  card.appendChild(head);
  if (open) {
    const holder = el("div", { class: "layer-body", "data-layer-controls": "right" });
    body(holder);
    card.appendChild(holder);
  }
  return card;
}

function cellOpacityRow(layer, holder) {
  const opacity = el("input", { type: "range", min: "0", max: "100", value: String(Math.round(layer.opacity * 100)), "aria-label": "Opacity" });
  opacity.addEventListener("input", () => setOpacity(layer, opacity.value / 100));
  holder.appendChild(cellControlRow("Opacity",[opacity]));
}

// A colour row's settings: the groups (click to hide), or the scale with its colours and the range of cells shown.
function cellColourBody(layer, column) {
  return (holder) => {
    if (layer.color_by !== column || layer.live) {
      holder.appendChild(el("p", { class: "hint", text: "Tick this row to colour the cells by " + cellsPreviewName(column) + "." }));
      return;
    }
    const scale = el("div", { hidden: true });
    holder.appendChild(scale);
    const colourMap = select([["viridis", "Viridis"], ["magma", "Magma"], ["inferno", "Inferno"], ["plasma", "Plasma"], ["cividis", "Cividis"],
      ["Reds", "Reds"], ["Blues", "Blues"]], layer.colormap || "viridis", (v) => { layer.colormap = v; bump(layer); });
    const [low, high] = layer.range || ["", ""];
    const from = el("input", { type: "number", step: "any", value: low, placeholder: "lowest", "aria-label": "Show cells from" });
    const to = el("input", { type: "number", step: "any", value: high, placeholder: "highest", "aria-label": "Show cells up to" });
    const apply = () => {
      claimCellView(layer,"Viewer filters");
      const a = from.value === "" ? "" : Number(from.value), b = to.value === "" ? "" : Number(to.value);
      layer.range = a === "" && b === "" ? null : [a, b];
      bump(layer);
    };
    from.addEventListener("change", apply); to.addEventListener("change", apply);
    scale.append(el('div',{class:'row cell-range'},[el('span',{class:'muted',text:'Colours'}),colourMap,
      el('span',{class:'muted',text:'Show range'}),from,el('span',{text:'–'}),to,
      help('This range filters cells and sets the colour span. Leave empty for every cell. It is not a biological positivity threshold.')]));
    legendInto(layer, holder, "/cells/" + layer.token + "/legend?" + cellColourQuery(layer),
      (kind) => { scale.hidden = kind !== "numeric"; });
  };
}

function cellGroupAppearance(layer) {
  const review = state.clusterReview;
  const shared=review && review.column === layer.color_by && review.table === dataView.combined?.path &&
    layer.table === layer.combinedTable ? review : null;
  const colours=layer.groupColours?.[JSON.stringify([layer.table,layer.color_by])];
  return colours?{...(shared||{}),colours:{...(shared?.colours||{}),...colours}}:shared;
}

function cellColourQuery(layer) {
  const [low, high] = layer.range || ["", ""];
  const appearance = cellGroupAppearance(layer);
  return "t=" + encodeURIComponent(layer.table || "") + "&col=" + encodeURIComponent(layer.color_by || "") +
    "&cmap=" + encodeURIComponent(layer.colormap || "viridis") + "&lo=" + encodeURIComponent(low) + "&hi=" + encodeURIComponent(high) +
    (appearance ? "&group_colours=" + encodeURIComponent(JSON.stringify(appearance.colours)) : "");
}

function activeReferenceOverlay(layer) {
  return !layer.color_by&&!layer.live&&!layer.flowSelection&&layer.referenceOverlayEnabled!==false&&
    layer.referenceOverlayTable===layer.table ? layer.referenceOverlay||'' : '';
}

function rememberReferenceOverlay(layer,column) {
  layer.referenceOverlay=column;layer.referenceOverlayTable=layer.table;
  refreshReferenceOverlayControls(layer);
}

function setCellColour(layer,column) {
  claimCellView(layer,'Viewer colours');
  if(layer.live)layer.referencePhenotypes=layer.live.phenotypes.slice();
  layer.live=null;layer.color_by=column||'';layer.hidden=new Set();layer.range=null;
  if(column&&(layer.mode||'outline')==='outline')layer.mode='both';
  if(!layer.visible)setVisible(layer,true);
  bump(layer);
}

function refreshReferenceOverlayControls(layer) {
  document.querySelectorAll('[data-reference-overlay]').forEach(box=>{if(box.layer===layer)box.refresh();});
}

function referenceOverlayControl(layer) {
  const box=el('div',{'data-reference-overlay':''});box.layer=layer;
  box.refresh=()=>{
    box.replaceChildren();box.hidden=!!layer.color_by||!!layer.live||!layer.referenceOverlay||layer.referenceOverlayTable!==layer.table;
    if(box.hidden)return;
    const on=el('input',{type:'checkbox','aria-label':'Colour assigned cells'});on.checked=layer.referenceOverlayEnabled!==false;
    on.onchange=()=>{layer.referenceOverlayEnabled=on.checked;bump(layer);};
    box.append(cellControlRow('Manual labels',[on,el('span',{class:'grow',text:layer.referenceOverlay})],
      'Colour assigned cells with their manual labels. Unassigned cells keep their outlines. Your cell filters still apply.'));
  };box.refresh();return box;
}

// The cell count of a layer, asked once.
function cellCountOf(layer) {
  if (layer.cellCount === undefined && !layer.pending) {
    layer.cellCount = null;
    getJSON("/cells/" + layer.token + "/ready").then((found) => {
      if (found.ready && found.cells !== null && found.cells !== undefined) { layer.cellCount = found.cells; buildPanel(); }
      else layer.cellCount = undefined;
    }).catch(() => { layer.cellCount = undefined; });
  }
  return layer.cellCount;
}

function cellBlock(layer, toolsOnly = false) {
  const blocks = state.annotationBlocksOpen || (state.annotationBlocksOpen = {});
  const title = layer.name || "Cells";
  const branch = el("details", { class: "layer-group annotation-subgroup", [toolsOnly ? "data-cell-source" : "data-cell-layer"]: layer.id });
  branch.open = blocks["cells::" + layer.id] ?? !toolsOnly;
  branch.addEventListener("toggle", () => { if (branch.isConnected) blocks["cells::" + layer.id] = branch.open; });
  const count = cellCountOf(layer);
  const drop = layer.added ? el("button", { class: "btn tiny", text: "×", "aria-label": "Remove " + title,
    title: layer.imported_cells ? "Take these imported cells off this image (the file stays)" : "Remove this layer from the view",
    onclick: async (event) => {
      event.preventDefault(); event.stopPropagation();
      if (layer.imported_cells || layer.data_layer) {
        if (!window.confirm("Take " + title + " off this image? " + "The source and generated files stay.")) return;
        try {
          if (layer.data_layer) await postJSON(cellDataRoute(), { action: "remove", image_token: layer.image_token, id: layer.data_layer });
          else await cellLayersPost({ action: "remove", layer_token: layer.token });
        } catch (error) { message(error.message, true); return; }
        removeLayer(layer, true);
      } else removeLayer(layer);
    } }) : null;
  const all = el("input", { type: "checkbox", "aria-label": "Show " + title, title: "Show or hide this layer's cells (rows keep their choices)" });
  all.checked = !!layer.visible;
  all.addEventListener("click", (event) => event.stopPropagation());
  all.addEventListener("change", () => { setVisible(layer, all.checked); buildPanel(); });
  branch.appendChild(el("summary", {}, [all, el("span", { class: "name", text: title, title: layer.source || title }),
    el("span", { class: "group-count", text: layer.pending ? "preparing…" : count ? Number(count).toLocaleString() + " cells" : "" }), drop].filter(Boolean)));
  if (layer.data_layer) {  // drawn from a cell data table: what it shows, and its circles or Voronoi cells to export
    const exportButton = el("button", { class: "btn tiny", text: "Export GeoJSON", title: "Download these circles or Voronoi cells as GeoJSON (each with its cell number)",
      onclick: () => { const link = el("a", { href: api(cellDataRoute("?image_token=" + encodeURIComponent(layer.image_token) + "&export=" + layer.data_layer)) }); document.body.appendChild(link); link.click(); link.remove(); } });
    // Size (circles) or reach (Voronoi), and Within: the same table drawn again.
    const rebuild = async (changes) => {
      message("Drawing again…");
      try {
        const made = await postJSON(cellDataRoute(), { action: "rebuild", image_token: layer.image_token, id: layer.data_layer, changes });
        const keep = { colourRows: layer.colourRows, color_by: layer.color_by, colormap: layer.colormap, mode: layer.mode, outline: layer.outline };
        removeLayer(layer, true);
        if (made.layer) await addLayerToView({ ...made.layer, ...keep, visible: true });
        buildPanel(); message(made.record.text + ".");
      } catch (error) { message(error.message, true); }
    };
    const number = (value, label, key) => {
      const input = el("input", { type: "number", min: "1", step: "1", value: String(value || ""), "aria-label": label, class: "cell-data-size" });
      input.addEventListener("change", () => { if (Number(input.value) > 0) rebuild({ [key]: Number(input.value) }); });
      return input;
    };
    const size = layer.data_show === "circles" ? [el("span", { class: "muted", text: "Size" }), number(layer.data_size, "Circle size µm", "size_um"), el("span", { class: "muted", text: "µm" })]
      : layer.data_show === "voronoi" ? [el("span", { class: "muted", text: "Reach" }), number(layer.data_reach, "Voronoi reach µm", "reach_um"), el("span", { class: "muted", text: "µm" })] : [];
    const within = el("select", { "aria-label": "Data within", title: "Keep only the rows inside an annotation" }, [el("option", { value: "", text: "Whole image" })]);
    within.addEventListener("focus", async () => {
      if (within.dataset.loaded) return; within.dataset.loaded = "1";
      try {
        const place = await getJSON(cellDataRoute("?image_token=" + encodeURIComponent(layer.image_token)));
        (place.areas || []).forEach(([group, items]) => within.appendChild(el("optgroup", { label: group }, items.map(([key, name]) => el("option", { value: key, text: name })))));
        within.value = layer.data_within || "";
      } catch (error) { message(error.message, true); }
    });
    if (layer.data_within) within.appendChild(el("option", { value: layer.data_within, text: (layer.data_text.match(/inside (.*)$/) || [])[1] || "annotation", selected: "" }));
    within.addEventListener("change", () => rebuild({ within: within.value }));
    branch.appendChild(el("div", { class: "row cell-data-line" }, [el("span", { class: "muted grow", text: layer.data_text || "" }), exportButton]));
    branch.appendChild(el("div", { class: "row cell-data-line" }, [...size, el("span", { class: "muted", text: "Within" }), within,
      help("Draw the same table again: circle size or Voronoi reach, and only the rows inside an annotation (cut at its edge).")]));
  }
  if (layer.selection) {
    branch.appendChild(el("p", { class: "hint", text: "Selected cells are highlighted. Refine or label them in the Phenotyping step." }));
    branch.appendChild(cellRow({ key: layer.id + "::selection", name: "Selected cells", shown: layer.visible,
      onShow: (on) => { setVisible(layer, on); buildPanel(); } }));
    return branch;
  }
  if(toolsOnly){branch.querySelector('summary').appendChild(el('button',{class:'btn tiny',
    text:activeCellLayer()===layer?'Active':'Use',disabled:activeCellLayer()===layer?'':null,
    'aria-label':'Use cell source '+title,onclick:event=>{event.preventDefault();event.stopPropagation();chooseCellSource(layer,layer.table,false);}}));}
  else branch.appendChild(cellSettings(layer));
  return branch;
}

// A cell layer's settings, one line each: the data its cells are coloured from (only tables holding these very
// cells), how they are drawn, what colours them, and which of them are shown.
function cellSettings(layer) {
  const box = el("div", { class: "cell-settings" });
  const line = cellControlRow;
  // Data
  const tables = layer.tables || [];
  const status=cellViewStatus(layer);
  if (!layer.combinedTable && layer.combinedLinkNote) box.appendChild(el('div',{class:'row'},[
    el('span',{class:'chip warn',text:layer.combinedLinkPending?'Preparing phenotyping results…':'Phenotyping results unavailable',title:layer.combinedLinkNote}),
    el('button',{class:'btn small',text:'Retry','aria-label':'Retry cell-table link',title:layer.combinedLinkNote,onclick:async event=>{
      event.currentTarget.disabled=true;
      try { await (state.phenotypingLink=dataRefreshCombined()); }
      catch(error){message(error.message,true);buildPanel();}
    }})
  ]));
  // Show
  const mode = select([["outline", "Outlines"], ["fill", "Filled"], ["both", "Filled + outlines"]], layer.mode || "outline", (v) => { layer.mode = v; bump(layer); });
  mode.setAttribute("aria-label", "Draw cells as");
  const opacity = el("input", { type: "range", min: "0", max: "100", value: String(Math.round((layer.opacity ?? 0.9) * 100)), "aria-label": "Cells opacity", class: "grow" });
  opacity.addEventListener("input", () => setOpacity(layer, opacity.value / 100));
  const outline = el("input", { type: "color", value: layer.outline || "#ffff00", "aria-label": "Outline colour", title: "Outline colour (and the cells' colour when nothing colours them)" });
  outline.addEventListener("change", () => { layer.outline = outline.value; bump(layer); });
  box.appendChild(line("Appearance", [mode, outline, opacity]));
  if (layer.flowSelection) {
    const population=layer.flowPopulation||{};
    box.appendChild(line('Flow gate',[
      el('span',{style:'display:inline-block;width:12px;height:12px;background:'+(population.colour||'#4e79a7')}),
      el('span',{text:(population.name||'Flow population')+' · '+Number(population.by_image?.[dataLayerRoi(layer)]||0).toLocaleString()+' cells'}),
      el('button',{class:'btn small',text:'Clear Flow selection',onclick:()=>window.clearFlowGateView?.()})
    ],'This population is linked to the Flow plots and Data. Clear restores ordinary cell colouring.'));
    box.append(line('Map',[densityFromCellsButton(layer)]));box.append(status);return box;
  }
  if (layer.selection) return box;
  box.appendChild(cellFilters(layer));
  // Colour by: a column of the data, as groups (text) or values (numbers)
  const colour = el("select", { "aria-label": "Colour the cells by" }, [el("option", { value: "", text: "Single colour" })]);
  if (layer.live) colour.appendChild(el("option", { value: "__live__", text: "Live RESOLVE phenotypes" }));
  if (layer.color_by && String(layer.color_by).startsWith("preview:")) colour.appendChild(el("option", { value: layer.color_by, text: cellsPreviewName(layer.color_by) }));
  colour.value = layer.live ? "__live__" : layer.color_by || "";
  if (tables.length) columnsFor(layer).then((all) => {
    const kinds = layer.columnKinds || {};
    const sorted = sortColumns(all).filter((c) => !/^(batch_id|label)$/.test(c));
    const groups = sorted.filter((c) => kinds[c] !== "value"), values = sorted.filter((c) => kinds[c] === "value");
    [["Groups", groups], ["Values", values]].forEach(([title, list]) => {
      if (list.length) colour.appendChild(el("optgroup", { label: title }, list.map((c) => el("option", { value: c, text: c }))));
    });
    colour.value = layer.live ? "__live__" : layer.color_by || "";
  }).catch(() => {});
  colour.addEventListener("change", () => {
    if (colour.value === "__live__") return;
    setCellColour(layer,colour.value);
  });
  box.appendChild(line("Colour by", [colour, densityFromCellsButton(layer)], "Groups (text, such as phenotypes) get a colour each: click one in the list below to hide or show it. " +
    "Values (numbers, such as a marker or a distance) get a colour scale, with the range it spans."));
  box.appendChild(referenceOverlayControl(layer));
  if (layer.color_by && !layer.live) {
    const legendBox = el("div", { class: "cell-legend" });
    box.appendChild(legendBox);
    cellColourBody(layer, layer.color_by)(legendBox);
  }
  const count=box.querySelector('.cell-filter-count');
  if(count)status.querySelector('.inline').prepend(count);
  box.appendChild(status);
  return box;
}

// Filters: only the cells passing all of them are drawn. A group column keeps the ticked groups; a value column a range.
function cellFilters(layer) {
  const holder = el("div", { class: "cell-filters" });
  const filters = layer.filters || (layer.filters = []);
  const route = (extra) => "/cells/" + layer.token + "/filter?" + cellColourQuery(layer) + '&hide='+encodeURIComponent(Array.from(layer.hidden||[]).join('\u001f')) + "&f=" +
    encodeURIComponent(JSON.stringify(filters)) + (extra || "");
  const chips = filters.map((item, index) => {
    const text = item.values ? item.col + ": " + item.values.slice(0, 3).join(", ") + (item.values.length > 3 ? " +" + (item.values.length - 3) : "")
      : item.col + " " + (item.lo ?? "…") + " to " + (item.hi ?? "…");
    return el("span", { class: "chip" }, [el("span", { text }), el("button", { class: "btn tiny ghost", text: "×", "aria-label": "Remove filter " + item.col,
      onclick: () => { claimCellView(layer,"Viewer filters"); filters.splice(index, 1); bump(layer); } })]);
  });
  const add = el("select", { "aria-label": "Add a filter" }, [el("option", { value: "", text: "+ Filter…" })]);
  if ((layer.tables || []).length) columnsFor(layer).then((all) => sortColumns(all).filter((c) => !/^(batch_id|label)$/.test(c))
    .forEach((c) => add.appendChild(el("option", { value: c, text: c })))).catch(() => {});
  const editor = el("div", { class: "cell-filter-editor" });
  add.addEventListener("change", async () => {
    const column = add.value; add.value = "";
    if (!column) return;
    editor.replaceChildren(el("span", { class: "muted", text: "Reading " + column + "…" }));
    try {
      const found = await getJSON(route("&values=" + encodeURIComponent(column)));
      if (!editor.isConnected) return;
      if (found.range) {
        const lo = el("input", { type: "number", step: "any", value: String(found.range[0]), "aria-label": "Filter from" });
        const hi = el("input", { type: "number", step: "any", value: String(found.range[1]), "aria-label": "Filter to" });
        editor.replaceChildren(el("span", { text: column }), lo, el("span", { text: "to" }), hi, el("button", { class: "btn small", text: "Apply",
          onclick: () => { claimCellView(layer,"Viewer filters"); filters.push({ col: column, lo: lo.value === "" ? null : Number(lo.value), hi: hi.value === "" ? null : Number(hi.value) }); bump(layer); } }));
      } else {
        const ticks = found.groups.map(([value, count]) => {
          const tick = el("input", { type: "checkbox", value, "aria-label": "Keep " + value }); tick.checked = true;
          return el("label", { class: "inline-check" }, [tick, el("span", { text: value + " (" + count.toLocaleString() + ")" })]);
        });
        const search=el('input',{type:'search',placeholder:'Find a group…','aria-label':'Find filter groups'});
        const matching=()=>ticks.filter(t=>t.textContent.toLocaleLowerCase().includes(search.value.toLocaleLowerCase()));
        const all=allBox('All matching',on=>{matching().forEach(t=>t.querySelector('input').checked=on);});
        const sync=()=>{ticks.forEach(t=>{t.hidden=!matching().includes(t);});all.sync(matching().map(t=>t.querySelector('input').checked));};
        ticks.forEach(t=>t.querySelector('input').addEventListener('change',sync));search.oninput=sync;sync();
        editor.replaceChildren(el('div',{class:'row grow'},[all.row,search]), el('div',{class:'cell-filter-groups'},ticks),
          el('button',{class:'btn small',text:'Apply',onclick:()=>{
            claimCellView(layer,'Viewer filters');filters.push({col:column,values:ticks.map(t=>t.querySelector('input')).filter(i=>i.checked).map(i=>i.value)});bump(layer);}}));
      }
    } catch (error) { editor.replaceChildren(el("span", { class: "hint", text: error.message })); }
  });
  const count = el("span", { class: "muted cell-filter-count" });
  if (layer.table) getJSON(route()).then((found) => { count.textContent = found.shown.toLocaleString() + " of " + found.total.toLocaleString() + " cells shown"; }).catch(() => {});
  holder.append(el("div", { class: "row field cell-line" }, [el("label", { class: "lbl", text: "Filters" }),
    el("div", { class: "inline grow wrap" }, [...chips, add, count]),
    help("Only the cells passing every filter are drawn (outlines too). Pick a column: tick the groups to keep, or set a range of values.")]), editor);
  return holder;
}

function densityBlock(maps) {
  const blocks = state.annotationBlocksOpen || (state.annotationBlocksOpen = {});
  const branch = el("div", { class: "cell-density-maps", "data-cell-layer": "density" });
  maps.forEach((layer) => {
    const single = String(layer.cmap || "").startsWith("#");
    const swatch = el("span", { class: "swatch density-swatch", style: "background:" + (single ? "linear-gradient(90deg,rgba(0,0,0,0)," + layer.cmap + ")" :
      "linear-gradient(90deg," + gradientStops(layer.cmap || "inferno") + ")") });
    branch.appendChild(cellRow({ key: layer.id, name: layer.name, shown: layer.visible, swatch, data: { "data-layer-id": layer.id },
      onShow: (on) => { setVisible(layer, on); buildPanel(); },
      onRemove: () => removeLayer(layer), removeTitle: "Remove this map",
      body: (holder) => {
        const run = () => densityControls(layer, holder);
        if (state.imagesReady) run(); else state.whenImagesReady.then(() => { if (holder.isConnected) run(); });
      } }));
  });
  return branch;
}

function cellLayersPost(body) {
  const image = state.layers.find((l) => l.type === "image");
  const session = new URLSearchParams(location.search).get("session");
  return postJSON("/studio/" + encodeURIComponent(session) + "/cell_layers", { image_token: image && image.token, ...body });
}

// Import another segmentation of this image as its own cell layer: one of the project's files, or a typed path.
function cellImportRow() {
  const image = state.layers.find((l) => l.type === "image");
  const session = new URLSearchParams(location.search).get("session");
  const pick = el("select", { "aria-label": "Import cells", title: "Show another segmentation of this image (its file is not changed)" });
  pick.appendChild(el("option", { value: "", text: "Add / switch cells…" }));
  const sources=state.layers.filter(l=>l.type==='cells'&&!l.selection&&!isTrainingCellLayer(l));
  const outlines=el('optgroup',{label:'Loaded outlines'}),tables=el('optgroup',{label:'Linked cell data'});
  sources.forEach(l=>{
    outlines.append(el('option',{value:'source:'+l.id,text:l.name||'Cells',title:l.source||l.name}));
    (l.tables||[]).forEach(t=>tables.append(el('option',{value:'table:'+l.id+'|'+t.id,text:cellDataInfo(l,t.id).name+' · '+l.name,title:t.file||t.name})));
  });
  if(outlines.children.length)pick.append(outlines);if(tables.children.length)pick.append(tables);
  const imports=el('optgroup',{label:'Imported tables · view in Data'});
  (dataView.opened||[]).forEach(item=>{
    const duplicates=dataView.opened.filter(other=>other.name===item.name).length>1;
    imports.append(el('option',{value:'file:'+item.file,text:item.name+(duplicates?' · '+item.file.slice(-6):''),title:item.path||item.name}));
  });
  if(imports.children.length)pick.append(imports);
  pick.addEventListener("focus", async () => {
    if (pick.dataset.loaded || !image) return;
    pick.dataset.loaded = "1";
    try {
      const found = await getJSON("/studio/" + encodeURIComponent(session) + "/cell_layers?image_token=" + encodeURIComponent(image.token));
      const files=el('optgroup',{label:'Add outlines from project'});
      (found.files || []).forEach(item=>files.append(el('option',{value:item.name,text:item.name,title:item.name})));pick.append(files);
      if (!(found.files || []).length) pick.appendChild(el("option", { value: "", text: "No other segmentation found", disabled: "" }));
    } catch (error) { message(error.message, true); }
  });
  const typed = el("input", { type: "text", placeholder: "or a GeoJSON path", "aria-label": "Cells GeoJSON path", class: "grow" });
  const table = el("input", { type: "file", accept: ".csv,.tsv,.parquet", hidden: "" });
  table.addEventListener("change", () => { if (table.files[0]) bringFiles([table.files[0]]); table.value = ""; });
  const tableButton = el("button", { class: "btn small", text: "Import cell data…", title: "Show a table of cells (CSV, TSV, Parquet) on the image: colour its cells, or draw circles or Voronoi cells. You can also drop it on the image.",
    onclick: () => table.click() });
  const add = async (body) => {
    try {
      const found = await cellLayersPost({ action: "add", ...body });
      if (!state.layers.includes(image)) return;
      await registerImportedCells(found.layer);
      (state.annotationBlocksOpen || (state.annotationBlocksOpen = {}))["cells::" + found.layer.id] = true;
      buildPanel(); message(found.layer.name + " added. Choose it in Add / switch cells to use it.");
    } catch (error) { message(error.message, true); }
  };
  pick.addEventListener('change',()=>{
    const value=pick.value;pick.value='';
    if(value.startsWith('source:')){const layer=sources.find(l=>l.id===value.slice(7));if(layer)chooseCellSource(layer,layer.table,false);}
    else if(value.startsWith('table:')){const [id,table]=value.slice(6).split('|');const layer=sources.find(l=>l.id===id);if(layer)chooseCellSource(layer,table);}
    else if(value.startsWith('file:')){const item=(dataView.opened||[]).find(item=>item.file===value.slice(5));if(item)dataOpenFile({file_token:item.file,name:item.name,path:item.path});}
    else if(value)add({name:value});
  });
  typed.addEventListener("keydown", (event) => { if (event.key === "Enter" && typed.value.trim()) add({ path: typed.value.trim() }); });
  return el("div", { class: "row cell-add cell-import" }, [pick, typed, table, tableButton,
    help("Add or explicitly switch to another segmentation and its linked table. Importing keeps the current viewer source. Choose a file of this project (another run, Tab 5's) or a GeoJSON path. " +
      "It comes back when the image opens again; its × takes it off (the file stays). The image's own cells are always the Segmentation step's latest.")]);
}

// Cells is one workspace: inspection, one active source, colours and maps.
function cellWorkspaceHeader(group) {
  const layer=activeCellLayer(), summary=group.querySelector('summary');
  const tools=el('div',{class:'cell-header-tools'});
  if(layer){
    const show=el('input',{type:'checkbox','aria-label':'Show cell outlines',title:'Show or hide cells'});show.checked=!!layer.visible;
    show.addEventListener('click',event=>event.stopPropagation());
    show.addEventListener('keydown',event=>event.stopPropagation());
    show.onchange=()=>{setVisible(layer,show.checked);buildPanel();};
    summary.querySelector('.name').before(show);
    const count=cellCountOf(layer);summary.querySelector('.group-count').textContent=count==null?(layer.pending?'Preparing…':''):Number(count).toLocaleString()+' cells';
  }
  const markers=cellMarkerControls(true);tools.append(markers.header);
  tools.addEventListener('click',event=>event.stopPropagation());tools.addEventListener('keydown',event=>event.stopPropagation());
  summary.querySelector('.name').after(tools);group.append(markers);
}

function cellSourceControl(layer) {
  const box=el('div',{class:'cell-sources'}),blocks=state.annotationBlocksOpen||(state.annotationBlocksOpen={});
  const sources=state.layers.filter(l=>l.type==='cells'&&!l.selection&&!isTrainingCellLayer(l));
  const pick=select(sources.map(l=>[l.id,l.name||'Cells']),layer.id,id=>{const next=sources.find(l=>l.id===id);if(next)chooseCellSource(next,next.table,false);});
  pick.setAttribute('aria-label','Cell outlines');
  sources.forEach((l,i)=>{pick.options[i].title=l.source||l.data_text||l.name;});
  const details=el('div',{class:'cell-source-details',hidden:blocks['cells::sources']?null:''});
  const fill=()=>{details.replaceChildren(...sources.map(l=>cellBlock(l,true)));};
  if(!details.hidden)fill();
  const button=el('button',{class:'btn small',text:'Sources…','aria-expanded':String(!details.hidden),onclick:()=>{
    details.hidden=!details.hidden;blocks['cells::sources']=!details.hidden;button.setAttribute('aria-expanded',String(!details.hidden));
    if(!details.hidden)fill();saveView();
  }});
  box.append(cellControlRow('Outlines',[pick,button],
    'Cell geometry, separate from the Cell data table. Sources keeps each segmentation visible independently and provides imported circles/Voronoi geometry and export controls. Removing a source only removes it from the view; its file stays.'),details);
  return box;
}

function cellBlocks(group, shown) {
  group.appendChild(sharedCellDataControl(state.layers));
  const current=activeCellLayer();
  if(current){group.append(cellSourceControl(current));const body=el('div',{'data-cell-layer':current.id,class:'cell-active-source'});body.append(cellSettings(current));group.append(body);}
  if (trainingCellsAvailable() && (!state.layersShownOnly || trainingCellsVisible())) group.appendChild(trainingCellsBlock());
  const maps = shown.filter((layer) => layer.type === 'density');
  if (maps.length) group.appendChild(densityBlock(maps));
  if (state.layers.some((l) => l.type === 'image')) group.appendChild(cellImportRow());
}
