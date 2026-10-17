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
  holder.appendChild(el("div", { class: "row" }, [el("label", { class: "lbl", text: "Opacity" }), opacity]));
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
      const a = from.value === "" ? "" : Number(from.value), b = to.value === "" ? "" : Number(to.value);
      layer.range = a === "" && b === "" ? null : [a, b];
      bump(layer);
    };
    from.addEventListener("change", apply); to.addEventListener("change", apply);
    scale.append(el("div", { class: "row" }, [el("label", { class: "lbl", text: "Colours" }), colourMap]),
      el("div", { class: "row field cell-range" }, [el("label", { class: "lbl", text: "Show" }), from, el("span", { text: "to" }), to,
        help("Colour only the cells with values in this range; the colours then span it. Leave both empty for every cell.")]));
    legendInto(layer, holder, "/cells/" + layer.token + "/legend?" + cellColourQuery(layer),
      (kind) => { scale.hidden = kind !== "numeric"; });
  };
}

function cellColourQuery(layer) {
  const [low, high] = layer.range || ["", ""];
  return "t=" + encodeURIComponent(layer.table || "") + "&col=" + encodeURIComponent(layer.color_by || "") +
    "&cmap=" + encodeURIComponent(layer.colormap || "viridis") + "&lo=" + encodeURIComponent(low) + "&hi=" + encodeURIComponent(high);
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

function cellBlock(layer) {
  const blocks = state.annotationBlocksOpen || (state.annotationBlocksOpen = {});
  const title = layer.name || "Cells";
  const branch = el("details", { class: "layer-group annotation-subgroup", "data-cell-layer": layer.id });
  branch.open = blocks["cells::" + layer.id] ?? true;
  branch.addEventListener("toggle", () => { if (branch.isConnected) blocks["cells::" + layer.id] = branch.open; });
  const count = cellCountOf(layer);
  const drop = layer.added ? el("button", { class: "btn tiny", text: "×", "aria-label": "Remove " + title,
    title: layer.imported_cells ? "Take these imported cells off this image (the file stays)" : "Remove this layer from the view",
    onclick: async (event) => {
      event.preventDefault(); event.stopPropagation();
      if (layer.imported_cells || layer.data_layer) {
        if (!window.confirm("Take " + title + " off this image? " + (layer.data_layer ? "The table itself stays; what was drawn from it goes." : "The file itself stays."))) return;
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
  branch.appendChild(cellSettings(layer));
  return branch;
}

// A cell layer's settings, one line each: the data its cells are coloured from (only tables holding these very
// cells), how they are drawn, what colours them, and which of them are shown.
function cellSettings(layer) {
  const box = el("div", { class: "cell-settings" });
  const line = (label, parts, tip) => el("div", { class: "row field cell-line" }, [el("label", { class: "lbl", text: label }),
    el("div", { class: "inline grow" }, parts.filter(Boolean)), tip ? help(tip) : null].filter(Boolean));
  // Data
  const tables = layer.tables || [];
  const hidden = layer.hidden_tables || [];
  const data = tables.length ? select(tables.map((t) => [t.id, t.name]), layer.table, (v) => {
    layer.table = v; layer.color_by = ""; layer.columns = null; layer.columnKinds = null; layer.range = null; layer.filters = [];
    if (layer.mode !== "outline") layer.mode = "outline";
    bump(layer);
  }) : el("span", { class: "muted", text: "No table for these cells" });
  if (tables.length) data.setAttribute("aria-label", "Cell data");
  const note = hidden.length ? el("span", { class: "chip warn", text: hidden.length + " hidden",
    title: hidden.map((h) => h.name + ": " + h.reason).join("\n") + "\nThese tables were made from another segmentation, so their values would land on the wrong cells." }) : null;
  box.appendChild(line("Data", [data, note], "The table whose columns colour and filter these cells. Only tables holding these very cells are offered " +
    "(the same cell numbers at the same places); a table made from another segmentation is left out, and the chip says why."));
  if (!layer.combinedTable && layer.combinedLinkNote) box.appendChild(el('span',{class:'chip warn',text:'Combined table not linked',title:layer.combinedLinkNote + '. Check Column roles in Phenotyping.'}));
  // Show
  const mode = select([["outline", "Outlines"], ["fill", "Filled"], ["both", "Filled + outlines"]], layer.mode || "outline", (v) => { layer.mode = v; bump(layer); });
  mode.setAttribute("aria-label", "Draw cells as");
  const opacity = el("input", { type: "range", min: "0", max: "100", value: String(Math.round((layer.opacity ?? 0.9) * 100)), "aria-label": "Cells opacity", class: "grow" });
  opacity.addEventListener("input", () => setOpacity(layer, opacity.value / 100));
  const outline = el("input", { type: "color", value: layer.outline || "#ffff00", "aria-label": "Outline colour", title: "Outline colour (and the cells' colour when nothing colours them)" });
  outline.addEventListener("change", () => { layer.outline = outline.value; bump(layer); });
  box.appendChild(line("Show", [mode, outline, opacity]));
  if (layer.selection) return box;
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
    if (layer.live) { layer.referencePhenotypes = layer.live.phenotypes.slice(); layer.live = null; }
    layer.color_by = colour.value; layer.hidden = new Set(); layer.range = null;
    if (colour.value && (layer.mode || "outline") === "outline") layer.mode = "both";  // coloured cells are filled
    bump(layer);
  });
  box.appendChild(line("Colour by", [colour], "Groups (text, such as phenotypes) get a colour each: click one in the list below to hide or show it. " +
    "Values (numbers, such as a marker or a distance) get a colour scale, with the range it spans."));
  if (layer.color_by && !layer.live) {
    const legendBox = el("div", { class: "cell-legend" });
    box.appendChild(legendBox);
    cellColourBody(layer, layer.color_by)(legendBox);
  }
  // Filter: which cells are drawn at all
  box.appendChild(cellFilters(layer));
  return box;
}

// Filters: only the cells passing all of them are drawn. A group column keeps the ticked groups; a value column a range.
function cellFilters(layer) {
  const holder = el("div", { class: "cell-filters" });
  const filters = layer.filters || (layer.filters = []);
  const route = (extra) => "/cells/" + layer.token + "/filter?t=" + encodeURIComponent(layer.table || "") + "&f=" +
    encodeURIComponent(JSON.stringify(filters)) + (extra || "");
  const chips = filters.map((item, index) => {
    const text = item.values ? item.col + ": " + item.values.slice(0, 3).join(", ") + (item.values.length > 3 ? " +" + (item.values.length - 3) : "")
      : item.col + " " + (item.lo ?? "…") + " to " + (item.hi ?? "…");
    return el("span", { class: "chip" }, [el("span", { text }), el("button", { class: "btn tiny ghost", text: "×", "aria-label": "Remove filter " + item.col,
      onclick: () => { filters.splice(index, 1); bump(layer); } })]);
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
      if (found.range) {
        const lo = el("input", { type: "number", step: "any", value: String(found.range[0]), "aria-label": "Filter from" });
        const hi = el("input", { type: "number", step: "any", value: String(found.range[1]), "aria-label": "Filter to" });
        editor.replaceChildren(el("span", { text: column }), lo, el("span", { text: "to" }), hi, el("button", { class: "btn small", text: "Apply",
          onclick: () => { filters.push({ col: column, lo: lo.value === "" ? null : Number(lo.value), hi: hi.value === "" ? null : Number(hi.value) }); bump(layer); } }));
      } else {
        const ticks = found.groups.map(([value, count]) => {
          const tick = el("input", { type: "checkbox", value, "aria-label": "Keep " + value }); tick.checked = true;
          return el("label", { class: "inline-check" }, [tick, el("span", { text: value + " (" + count.toLocaleString() + ")" })]);
        });
        editor.replaceChildren(el("div", { class: "muted", text: column + ": keep the ticked groups" }), el("div", { class: "cell-filter-groups" }, ticks),
          el("button", { class: "btn small", text: "Apply", onclick: () => {
            filters.push({ col: column, values: ticks.map((t) => t.querySelector("input")).filter((i) => i.checked).map((i) => i.value) }); bump(layer); } }));
      }
    } catch (error) { editor.replaceChildren(el("span", { class: "hint", text: error.message })); }
  });
  const count = el("span", { class: "muted" });
  if (filters.length) getJSON(route()).then((found) => { count.textContent = found.shown.toLocaleString() + " of " + found.total.toLocaleString() + " cells shown"; }).catch(() => {});
  holder.append(el("div", { class: "row field cell-line" }, [el("label", { class: "lbl", text: "Filter" }),
    el("div", { class: "inline grow wrap" }, [...chips, add, count]),
    help("Only the cells passing every filter are drawn (outlines too). Pick a column: tick the groups to keep, or set a range of values.")]), editor);
  return holder;
}

function densityBlock(maps) {
  const blocks = state.annotationBlocksOpen || (state.annotationBlocksOpen = {});
  const branch = el("details", { class: "layer-group annotation-subgroup", "data-cell-layer": "density" });
  branch.open = blocks["cells::density"] ?? true;
  branch.addEventListener("toggle", () => { if (branch.isConnected) blocks["cells::density"] = branch.open; });
  branch.appendChild(el("summary", {}, [el("span", { class: "name", text: "Density maps" }),
    el("span", { class: "group-count", text: maps.length + (maps.length === 1 ? " map" : " maps") })]));
  maps.forEach((layer) => {
    const single = String(layer.cmap || "").startsWith("#");
    const swatch = el("span", { class: "swatch density-swatch", style: "background:" + (single ? "linear-gradient(90deg,rgba(0,0,0,0)," + layer.cmap + ")" :
      "linear-gradient(90deg," + gradientStops(layer.cmap || "inferno") + ")") });
    branch.appendChild(cellRow({ key: layer.id, name: layer.name, shown: layer.visible, swatch, data: { "data-layer-id": layer.id },
      onShow: (on) => { setVisible(layer, on); buildPanel(); },
      onRemove: () => removeLayer(layer), removeTitle: "Remove this map",
      body: (holder) => {
        cellOpacityRow(layer, holder);
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
  pick.appendChild(el("option", { value: "", text: "Import cells…" }));
  pick.addEventListener("focus", async () => {
    if (pick.dataset.loaded || !image) return;
    pick.dataset.loaded = "1";
    try {
      const found = await getJSON("/studio/" + encodeURIComponent(session) + "/cell_layers?image_token=" + encodeURIComponent(image.token));
      (found.files || []).forEach((item) => pick.appendChild(el("option", { value: item.name, text: item.name })));
      if (!(found.files || []).length) pick.appendChild(el("option", { value: "", text: "No other segmentation found", disabled: "" }));
    } catch (error) { message(error.message, true); }
  });
  const typed = el("input", { type: "text", placeholder: "or a GeoJSON path", "aria-label": "Cells GeoJSON path", class: "grow" });
  const table = el("input", { type: "file", accept: ".csv,.tsv,.parquet", hidden: "" });
  table.addEventListener("change", () => { if (table.files[0]) bringFiles([table.files[0]]); table.value = ""; });
  const tableButton = el("button", { class: "btn small", text: "Cell data…", title: "Show a table of cells (CSV, TSV, Parquet) on the image: colour its cells, or draw circles or Voronoi cells. You can also drop it on the image.",
    onclick: () => table.click() });
  const add = async (body) => {
    try {
      const found = await cellLayersPost({ action: "add", ...body });
      if (!state.layers.includes(image)) return;
      await addLayerToView(found.layer);
      (state.annotationBlocksOpen || (state.annotationBlocksOpen = {}))["cells::" + found.layer.id] = true;
      buildPanel(); message(found.layer.name + " added; its × takes it off again.");
    } catch (error) { message(error.message, true); }
  };
  pick.addEventListener("change", () => { if (pick.value) add({ name: pick.value }); pick.value = ""; });
  typed.addEventListener("keydown", (event) => { if (event.key === "Enter" && typed.value.trim()) add({ path: typed.value.trim() }); });
  return el("div", { class: "row cell-add cell-import" }, [pick, typed, table, tableButton,
    help("Show another segmentation of this image as its own cell layer: a file of this project (another run, Tab 5's) or a GeoJSON path. " +
      "It comes back when the image opens again; its × takes it off (the file stays). The image's own cells are always the Segmentation step's latest.")]);
}

// The Cells and maps group of Layers in Studio.
function cellBlocks(group, shown) {
  shown.filter((layer) => layer.type === "cells").forEach((layer) => group.appendChild(cellBlock(layer)));
  const maps = shown.filter((layer) => layer.type === "density");
  if (maps.length) group.appendChild(densityBlock(maps));
  if (state.layers.some((l) => l.type === "image")) group.appendChild(cellImportRow());
}
