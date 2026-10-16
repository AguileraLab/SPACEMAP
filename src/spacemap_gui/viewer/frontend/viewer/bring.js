// Files brought in by hand, in Studio: dropped on the image, or typed as a path (a remote node's folder, say).
// A GeoJSON becomes an editable annotation layer (the file is not changed). An image opens, or, when it has the same
// size as the open one, can be laid over it as its own layer (on/off, opacity, markers) instead.
"use strict";

function bringRoute(action) {
  const session = new URLSearchParams(location.search).get("session");
  return "/studio/" + encodeURIComponent(session) + "/bring/" + action;
}

// One answer from the server (a GeoJSON's polygons, or an image's facts) put to use.
async function bringAnswer(found) {
  if (found.kind === "geojson") { await importAnnotationCopy({ name: found.name, features: found.features }); return; }
  if (found.kind === "table") { await cellDataDialog(found); return; }
  if (found.kind === "cells") {  // numbered cell outlines: as cells (a segmentation) unless asked otherwise
    const choice = await askChoice(found.name, found.count.toLocaleString() + " numbered polygons: they look like cell outlines (a segmentation).",
      [["cells", "Add as cells", true], ["annotations", "Add as annotations"]]);
    if (choice === "cells") {
      const made = await cellLayersPost({ action: "add", file_token: found.file_token });
      await addLayerToView(made.layer); buildPanel(); message(made.layer.name + " added to Cells and maps; its × takes it off.");
    } else if (choice === "annotations") {
      const shapes = await postJSON(bringRoute("shapes"), { file_token: found.file_token });
      await importAnnotationCopy({ name: shapes.name, features: shapes.features });
    }
    return;
  }
  const base = state.base && state.base.info;
  const same = base && base.width === found.width && base.height === found.height;
  const choice = await bringChoice(found, same);
  if (choice === "overlay") {
    const made = await postJSON(bringRoute("overlay"), { image_token: found.image_token });
    await addLayerToView({ ...made.layer, visible: true });
    buildPanel(); message(found.name + " is laid over the image: see Layers (tick, opacity, markers; × removes it).");
  } else if (choice === "open") {
    const session = new URLSearchParams(location.search).get("session");
    const added = await postJSON(bringRoute("open"), { image_token: found.image_token });
    const result = await postJSON("/session/" + encodeURIComponent(session) + "/open", { id: added.added });
    state.sessionVersion = result.version;
    await init(result.spec);
    message(found.name + " is open (it is in the image list now).");
  }
}

// A dialog with a title, a line and buttons [[value, label, primary]]; resolves with the value (null: cancelled).
function askChoice(title, text, buttons) {
  return new Promise((resolve) => {
    $("bring-dialog")?.close(); $("bring-dialog")?.remove();
    const dialog = el("dialog", { id: "bring-dialog", class: "annotation-import-dialog", "aria-labelledby": "bring-title" });
    const done = (value) => { dialog.close(); dialog.remove(); resolve(value); };
    dialog.append(el("h3", { id: "bring-title", text: title }), el("p", { class: "muted", text }),
      el("div", { class: "row" }, [...buttons.map(([value, label, primary]) => el("button", { class: "btn" + (primary ? " primary" : ""), text: label, onclick: () => done(value) })),
        el("button", { class: "btn ghost", text: "Cancel", onclick: () => done(null) })]));
    dialog.addEventListener("cancel", () => done(null));
    document.body.appendChild(dialog); dialog.showModal();
  });
}

// Overlay or open: a small dialog; overlay only when the sizes match.
function bringChoice(found, same) {
  return new Promise((resolve) => {
    $("bring-dialog")?.close();
    const dialog = el("dialog", { id: "bring-dialog", class: "annotation-import-dialog", "aria-labelledby": "bring-title" });
    const done = (value) => { dialog.close(); dialog.remove(); resolve(value); };
    const size = found.width.toLocaleString() + " × " + found.height.toLocaleString() + " px";
    dialog.append(el("h3", { id: "bring-title", text: found.name }),
      el("p", { class: "muted", text: size + (same ? ", the same size as the open image." : state.base ? ", not the size of the open image, so it can only be opened." : ".") }),
      el("div", { class: "row" }, [
        same ? el("button", { class: "btn primary", text: "Lay over this image", onclick: () => done("overlay") }) : null,
        el("button", { class: "btn" + (same ? "" : " primary"), text: "Open it instead", onclick: () => done("open") }),
        el("button", { class: "btn ghost", text: "Cancel", onclick: () => done(null) })].filter(Boolean)));
    dialog.addEventListener("cancel", () => done(null));
    document.body.appendChild(dialog);
    dialog.showModal();
  });
}

// A dropped file goes to the project (outputs/studio/dropped), with its upload shown as it goes.
function bringUpload(file) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", api(bringRoute("drop") + "?name=" + encodeURIComponent(file.name)));
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) message("Bringing in " + file.name + ": " + Math.round(100 * event.loaded / event.total) + "%");
    };
    request.onload = () => {
      let found = {};
      try { found = JSON.parse(request.responseText); } catch (_) { /* below */ }
      if (request.status >= 200 && request.status < 300) resolve(found);
      else reject(new Error(found.error || "Could not bring in " + file.name + "."));
    };
    request.onerror = () => reject(new Error("Could not bring in " + file.name + "."));
    request.send(file);
  });
}

// Several GeoJSON files dropped at once in Studio: each goes to the image its name contains (P11_roi_1_old.geojson to
// P11_roi_1), so every image's Layers shows only its own file. One file still goes to the open image.
async function bringMatchedGeoJSON(files) {
  const session = new URLSearchParams(location.search).get("session");
  const matched = [], unmatched = [], waiting = [];
  for (const [number, file] of files.entries()) {
    message("Matching " + file.name + " to its image (" + (number + 1) + " of " + files.length + ")…");
    try {
      const raw = JSON.parse(await file.text());
      const features = Array.isArray(raw) ? raw : raw.type === "FeatureCollection" ? raw.features : raw.type === "Feature" ? [raw] : null;
      if (!features) throw Error(file.name + " has no polygons.");
      const found = await postJSON("/studio/" + encodeURIComponent(session) + "/annotation_export", { action: "import_matched", name: file.name, features });
      if (!found.matched) { unmatched.push(file.name); waiting.push({ name: file.name, features, images: found.images || [] }); continue; }
      matched.push(file.name + " → " + found.matched.replace(/(\.ome)?\.(tiff?|qptiff)$/i, "") + (found.already ? " (already imported)" : ""));
      if (found.open && found.layer) { await addLayerToView({ ...found.layer, open: true }); (state.layerGroups || (state.layerGroups = {})).annotations = true; }
    } catch (error) { unmatched.push(file.name + " (" + error.message + ")"); }
  }
  // Files matching no image wait in Export > Imported files, where their image is chosen.
  const named = new Set(waiting.map((item) => item.name));
  state.unmatchedImports = [...(state.unmatchedImports || []).filter((item) => !named.has(item.name)), ...waiting];
  buildPanel();
  document.dispatchEvent(new CustomEvent("spacemap-annotations-changed", { detail: { layer: null } }));
  document.dispatchEvent(new CustomEvent("spacemap-imports-unmatched"));
  message((matched.length ? "Imported " + matched.length + ": " + matched.join(", ") + "." : "") +
    (unmatched.length ? " Matching no image: " + unmatched.join(", ") + ". Choose their images in Annotation > Export > Imported files." : ""), !matched.length);
  return { matched, unmatched };
}

// Cell outlines dropped with their cell table (P11_roi_1.ome_cell_boundaries.geojson and P11_roi_1.ome_cell_features.csv):
// one cells layer whose data is that table.
function cellsName(name) {
  return String(name).toLowerCase().replace(/(_cell_boundaries|_cell_features|_cells|_boundaries|_features)?\.(geojson|json|csv|tsv|parquet)$/, "");
}
async function bringCellPairs(files) {
  const tables = files.filter((file) => /\.(csv|tsv|parquet)$/i.test(file.name));
  const used = new Set();
  for (const outlines of files.filter((file) => /\.geojson$/i.test(file.name))) {
    const table = tables.find((item) => !used.has(item) && cellsName(item.name) === cellsName(outlines.name));
    if (!table) continue;
    const head = await outlines.slice(0, 262144).text();
    if (!/"(label|cell_label)"\s*:\s*\d/.test(head)) continue;  // not numbered cell outlines
    try {
      message("Bringing in " + outlines.name + " with " + table.name + "…");
      const [shapes, data] = [await bringUpload(outlines), await bringUpload(table)];
      const made = await cellLayersPost({ action: "add", file_token: shapes.file_token, table_token: data.file_token });
      await addLayerToView(made.layer); buildPanel();
      message(made.layer.name + " added to Cells and maps with " + table.name + " as its data; its × takes it off.");
      used.add(outlines); used.add(table);
    } catch (error) { message(error.message, true); }
  }
  return files.filter((file) => !used.has(file));
}

async function bringFiles(files) {
  files = await bringCellPairs(files);
  const geojson = files.filter((file) => /\.(geo)?json$/i.test(file.name));
  if (state.studio && geojson.length > 1) {
    await bringMatchedGeoJSON(geojson);
    files = files.filter((file) => !geojson.includes(file));
  }
  for (const file of files) {
    const name = file.name.toLowerCase();
    try {
      if (name.endsWith(".geojson") || name.endsWith(".json")) {
        const raw = JSON.parse(await file.text());
        const features = Array.isArray(raw) ? raw : raw.type === "FeatureCollection" ? raw.features : raw.type === "Feature" ? [raw] : null;
        if (!features) throw Error(file.name + " has no polygons.");
        const numbered = features.slice(0, 2000).filter((f) => typeof (f.properties || {}).label === "number" || typeof (f.properties || {}).cell_label === "number").length;
        if (features.length >= 50 && numbered >= 0.9 * Math.min(features.length, 2000)) await bringAnswer(await bringUpload(file));  // cell outlines
        else await importAnnotationCopy({ name: file.name, features });
      } else if (/\.(csv|tsv|parquet)$/.test(name)) {
        await bringAnswer(await bringUpload(file));  // cell data
      } else if (/\.(qptiff|ome\.tiff?|tiff?|svs|png|jpe?g|bmp)$/.test(name)) {
        await bringAnswer(await bringUpload(file));
      } else message(file.name + ": drop a GeoJSON, a cell table (CSV, TSV, Parquet) or an image (QPTIFF, OME-TIFF, TIFF, SVS, PNG, JPG).", true);
    } catch (error) { message(error.message, true); }
  }
}

// Drop files on the image.
function setUpDrop() {
  const target = $("osd");
  if (!target || target.dataset.dropReady) return;
  target.dataset.dropReady = "1";
  target.addEventListener("dragover", (event) => {
    if (!state.studio || !Array.from(event.dataTransfer?.types || []).includes("Files")) return;
    event.preventDefault(); target.classList.add("drop-over");
  });
  target.addEventListener("dragleave", () => target.classList.remove("drop-over"));
  target.addEventListener("drop", (event) => {
    if (!state.studio || !event.dataTransfer?.files?.length) return;
    event.preventDefault(); target.classList.remove("drop-over");
    bringFiles(Array.from(event.dataTransfer.files));
  });
}

// A typed path: a file opens; a folder lists its GeoJSON files and images to pick from.
function bringPathRow() {
  const input = el("input", { type: "text", class: "grow", placeholder: "Path of a file or folder…", "aria-label": "Bring in a path" });
  const pick = el("select", { "aria-label": "File in the folder", hidden: "" });
  let folder = null;
  const ask = async (body) => {
    try {
      const found = await postJSON(bringRoute("path"), body);
      if (found.folder) {
        folder = found.folder;
        pick.replaceChildren(el("option", { value: "", text: found.files.length ? found.files.length + " files: choose one…" : "No GeoJSON or image here" }),
          ...found.files.map((name) => el("option", { value: name, text: name })));
        pick.hidden = false;
        return;
      }
      await bringAnswer(found);
    } catch (error) { message(error.message, true); }
  };
  input.addEventListener("keydown", (event) => { if (event.key === "Enter" && input.value.trim()) { pick.hidden = true; ask({ path: input.value.trim() }); } });
  pick.addEventListener("change", () => { if (pick.value) ask({ path: folder, name: pick.value }); });
  return el("div", { class: "row wrap bring-path" }, [input, pick,
    help("Type the full path of a GeoJSON, a cell table (CSV, TSV, Parquet), an image (QPTIFF, OME-TIFF, TIFF, SVS, PNG, JPG) or a folder holding them, on the computer " +
      "running SPACEMAP (a remote node too), and press Enter. You can also drop files on the image. A GeoJSON becomes an editable " +
      "annotation layer; an image of the same size can be laid over the open one, or opened. Files are never changed.")]);
}

// ---------------------------------------------------------------- cell data
function cellDataRoute(query) {
  const session = new URLSearchParams(location.search).get("session");
  return "/studio/" + encodeURIComponent(session) + "/cell_data" + (query || "");
}

// Set a table up: which columns hold x and y (and their units), which rows are this image's, how to show the cells,
// and inside which annotation. Columns, units and the image's rows are guessed; everything can be changed.
async function cellDataDialog(found) {
  const image = state.layers.find((l) => l.type === "image");
  if (!image) { message("Open an image first, then bring the table in.", true); return; }
  message("Reading " + found.name + "…");
  const [facts, place] = await Promise.all([
    postJSON(cellDataRoute(), { action: "inspect", image_token: image.token, file_token: found.file_token }),
    getJSON(cellDataRoute("?image_token=" + encodeURIComponent(image.token)))]);
  $("bring-dialog")?.close(); $("bring-dialog")?.remove();
  const dialog = el("dialog", { id: "bring-dialog", class: "annotation-import-dialog cell-data-dialog", "aria-labelledby": "bring-title" });
  const pick = (label, options, value) => {
    const select = el("select", { "aria-label": label });
    options.forEach(([key, text]) => select.appendChild(el("option", { value: key, text })));
    select.value = value; return select;
  };
  const columns = facts.columns.map((c) => [c, c]);
  const x = pick("X column", columns, facts.x), y = pick("Y column", columns, facts.y);
  const units = pick("Position units", [["px", "pixels"], ["um", "µm"]], facts.units);
  const roiColumn = pick("Column naming the image", [["", "(every row is this image's)"], ...columns], facts.roi_column);
  const roiValue = pick("This image's value", facts.roi_values.map((v) => [v, v]), facts.roi_value || facts.roi_values[0] || "");
  const label = pick("Cell number column", [["", "(none)"], ...columns], facts.label_column);
  const show = pick("Show as", [
    ...(place.segmentation ? [["cells:label", "This image's cells, by cell number"], ["cells:position", "This image's cells, by position"]] : []),
    ["circles", "Circles"], ["voronoi", "Voronoi cells"]], place.segmentation ? (facts.label_column ? "cells:label" : "cells:position") : "voronoi");
  const size = el("input", { type: "number", min: "1", step: "1", value: "10", "aria-label": "Circle size µm" });
  const reach = el("input", { type: "number", min: "1", step: "1", value: "15", "aria-label": "Voronoi reach µm" });
  const within = el("select", { "aria-label": "Within" }, [el("option", { value: "", text: "Whole image" })]);
  (place.areas || []).forEach(([group, items]) => within.appendChild(el("optgroup", { label: group }, items.map(([key, name]) => el("option", { value: key, text: name })))));
  const line = (text, control, tip) => el("div", { class: "row field" }, [el("label", { class: "lbl", text }), control, tip ? help(tip) : null].filter(Boolean));
  const sizeRow = line("Circle size", el("span", { class: "inline" }, [size, el("span", { class: "muted", text: "µm across" })]));
  const reachRow = line("Reach", el("span", { class: "inline" }, [reach, el("span", { class: "muted", text: "µm at most" })]),
    "A Voronoi cell is the area nearer to its point than to any other, cut at this distance so lone cells stay round.");
  const labelRow = line("Cell number", label, "The column with the segmentation's cell numbers (label).");
  const roiRow = line("This image", roiValue);
  const sync = () => {
    sizeRow.hidden = show.value !== "circles"; reachRow.hidden = show.value !== "voronoi";
    labelRow.hidden = show.value !== "cells:label"; roiRow.hidden = !roiColumn.value;
  };
  [show, roiColumn].forEach((s) => s.addEventListener("change", sync));
  const status = el("p", { class: "muted", role: "status" });
  const build = el("button", { class: "btn primary", text: "Show on the image", onclick: async () => {
    build.disabled = true; status.textContent = "Building…";
    const [kind, link] = show.value.split(":");
    const choice = { x: x.value, y: y.value, units: units.value, roi_column: roiColumn.value, roi_value: roiColumn.value ? roiValue.value : null,
      label_column: label.value, show: kind, link: link || "", size_um: Number(size.value) || 10, reach_um: Number(reach.value) || 15, within: within.value };
    try {
      const made = await postJSON(cellDataRoute(), { action: "build", image_token: image.token, file_token: found.file_token, choice });
      if (!state.layers.includes(image)) return;
      if (made.layer) {
        const old = state.layers.find((l) => l.id === made.layer.id);
        if (old) removeLayer(old, true);
        const layer = await addLayerToView({ ...made.layer, visible: true });
        if (kind === "cells") { layer.table = made.layer.tables.find((t) => t.name === "Data: " + found.name)?.id || layer.table; layer.columns = null; }
        (state.layerGroups || (state.layerGroups = {})).cells = true;
        buildPanel(); refreshLayer(layer);
      }
      dialog.close(); dialog.remove();
      message(found.name + ": " + made.record.text + ". In Cells and maps: Colour by any of its columns; its × takes it off.");
    } catch (error) { status.textContent = error.message; build.disabled = false; }
  } });
  dialog.append(el("h3", { id: "bring-title", text: "Cell data: " + found.name }),
    el("p", { class: "muted", text: facts.columns.length + " columns. Check the guesses, then show it." }),
    line("X, Y", el("span", { class: "inline" }, [x, y, units]), "The columns with each cell's position, in pixels of this image or in µm (divided by its pixel size)."),
    line("Image column", roiColumn, "A table of several images: the column naming each row's image. Only this image's rows are shown."), roiRow,
    line("Show as", show, "This image's cells: the table colours the image's own cells, matched by cell number, or by the cell each position falls in (when the numbers come from another run). Circles: a disc per row. Voronoi cells: the area nearest each row, when there is no segmentation."),
    labelRow, sizeRow, reachRow,
    line("Within", within, "Keep only the rows inside one annotation (the circles and Voronoi cells are cut at its edge)."),
    el("div", { class: "row" }, [build, el("button", { class: "btn", text: "Open in the data viewer", title: "Look at the table's rows and columns (Data tab, below the image) without drawing anything",
      onclick: () => { dialog.close(); dialog.remove(); dataOpenFile(found); } }),
      el("button", { class: "btn ghost", text: "Cancel", onclick: () => { dialog.close(); dialog.remove(); } })]), status);
  dialog.addEventListener("cancel", () => dialog.remove());
  document.body.appendChild(dialog); sync(); dialog.showModal();
}
