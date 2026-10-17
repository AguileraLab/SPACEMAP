/* SPACEMAP viewer, part 9 of 12: marker sets.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- marker sets
// Named groups of markers ("T cells", "B cells") with their colours, ranges and
// names, saved for the project and applied by marker name on any image.
function markerSets(layer) {
  const box = el("div", { class: "sets" });
  const row = el("div", { class: "row wrap" }, [el("label", { class: "lbl", text: "Marker sets" })]);
  const pick = el("select", { title: "Show a saved set: its markers with their colours, ranges and names; the others are hidden" });
  pick.appendChild(el("option", { value: "", text: "Choose a set..." }));
  const remove = el("button", { class: "btn small", text: "Delete", title: "Delete the chosen set" });
  remove.hidden = true;
  const save = el("button", { class: "btn small", text: "Save ticked as set...", title: "Save the ticked markers, with their colours, ranges and names, under a name" });
  row.append(pick, remove, save);
  box.appendChild(row);
  const naming = el("div", { class: "row" });
  naming.hidden = true;
  const name = el("input", { type: "text", placeholder: "Set name, e.g. T cells" });
  const ok = el("button", { class: "btn small primary", text: "Save" });
  const cancel = el("button", { class: "btn small", text: "Cancel", onclick: () => { naming.hidden = true; } });
  naming.append(name, ok, cancel);
  box.appendChild(naming);
  let sets = {};
  const fill = (all) => {
    sets = all || {};
    layer.markerSets = sets;
    document.dispatchEvent(new CustomEvent('spacemap-marker-sets', {detail:{layer}}));
    const chosen = pick.value;
    pick.innerHTML = "";
    pick.appendChild(el("option", { value: "", text: "Choose a set..." }));
    Object.keys(sets).sort((a, b) => a.localeCompare(b)).forEach((key) => {
      const present = sets[key].filter((m) => layer.channels.some((c) => c.name === m.name)).length;
      pick.appendChild(el("option", { value: key, text: key + " (" + present + " of " + sets[key].length + " here)" }));
    });
    pick.value = sets[chosen] ? chosen : (sets[layer.activeSet] ? layer.activeSet : "");
    remove.hidden = !pick.value;
  };
  getJSON("/display/" + layer.display_token + "/sets").then((r) => fill(r.sets)).catch(() => {});
  pick.addEventListener("change", () => {
    remove.hidden = !pick.value;
    if (pick.value) applySet(layer, pick.value, sets[pick.value] || []);
  });
  remove.addEventListener("click", async () => {
    if (!pick.value) return;
    const gone = pick.value;
    try {
      fill((await postJSON("/display/" + layer.display_token + "/sets", { delete: gone })).sets);
      message("Deleted the set " + gone + ".");
    } catch (error) { message(error.message, true); }
  });
  save.addEventListener("click", () => { naming.hidden = false; name.value = layer.activeSet || ""; name.focus(); });
  const store = async () => {
    const markers = layer.channels.filter((c) => c.visible).map((c) => ({ name: c.name, label: c.label || c.name,
      color: c.color, low: c.low, high: c.high }));
    try {
      const result = await postJSON("/display/" + layer.display_token + "/sets", { name: name.value.trim(), markers: markers });
      layer.activeSet = name.value.trim();
      saveView();
      fill(result.sets);
      naming.hidden = true;
      message("Saved " + markers.length + " marker(s) as " + layer.activeSet + ".");
    } catch (error) { message(error.message, true); }
  };
  ok.addEventListener("click", store);
  name.addEventListener("keydown", (e) => { if (e.key === "Enter") store(); if (e.key === "Escape") naming.hidden = true; });
  return box;
}

async function applySet(layer, setName, markers) {
  const byName = {};
  markers.forEach((m) => { byName[m.name] = m; });
  if (!layer.channels.some((c) => byName[c.name])) {
    message("None of the markers in " + setName + " are in this image.", true);
    return;
  }
  const needRange = [];
  layer.channels.forEach((c) => {
    const m = byName[c.name];
    c.visible = !!m;
    if (!m) return;
    c.color = m.color; c.custom = true;  // a marker set's colours are the user's choice
    c.label = m.label || c.name;
    if (m.low !== null && m.low !== undefined) c.low = m.low;
    if (m.high !== null && m.high !== undefined) c.high = m.high;
    if (c.low === undefined || c.high === undefined) needRange.push(c);
  });
  await fillLimits(layer, needRange);
  const linkedCells = liveForImage(layer), changes = {};
  if (linkedCells) {
    layer.channels.forEach((channel) => {
      const marker = linkedMarker(layer, channel);
      if (marker && channel.resolveLinked !== false && Number.isFinite(channel.low)) {
        changes[marker.marker] = marker.threshold = channel.low; marker.source = "changed";
      }
    });
    if (Object.keys(changes).length) sendLive(linkedCells, changes);
  }
  layer.activeSet = setName;
  remember(layer);
  refreshLayer(layer);
  buildPanel();
  const shown = layer.channels.filter((c) => c.visible).length;
  message(setName + ": " + shown + " marker(s) shown" + (shown < markers.length ? " (" + (markers.length - shown) + " not in this image)" : "") + ".");
}

function channelRow(layer, channel) {
  const row = el("div", { class: "channel" + (channel.visible ? "" : " off") });
  const check = el("input", { type: "checkbox" });
  check.checked = channel.visible;
  const color = el("input", { type: "color", value: channel.color });
  const low = el("input", { type: "number", value: channel.low === undefined ? "" : channel.low, title: "Values at or below this are black" });
  const high = el("input", { type: "number", value: channel.high === undefined ? "" : channel.high, title: "Values at or above this are full colour" });
  const auto = el("button", { class: "btn small", text: "Auto", title: "Set the range from the image" });
  const changed = () => { remember(layer); refreshLayer(layer); };
  check.addEventListener("change", async () => {
    channel.visible = check.checked;
    row.classList.toggle("off", !channel.visible);
    if (channel.visible && (channel.low === undefined || channel.high === undefined)) {
      await fillLimits(layer, [channel]); low.value = channel.low; high.value = channel.high;
    }
    changed();
    if (layer.syncAllMarkers) layer.syncAllMarkers();
  });
  color.addEventListener("input", () => { channel.color = color.value; channel.custom = true; changed(); });
  low.step = high.step = "any";
  low.addEventListener("change", () => {
    if (!setMarkerRange(layer, channel, parseFloat(low.value), markerHigh(channel, parseFloat(low.value)))) low.value = channel.low;
  });
  high.addEventListener("change", () => {
    if (!setMarkerRange(layer, channel, channel.low, parseFloat(high.value))) high.value = channel.high;
  });
  auto.title = "Auto display range: lower via " + (layer.autoMethod || "Triangle") + "; upper via P" + (layer.displayAuto?.upper_percentile ?? 99.8);
  auto.disabled = !!layer.autoRun;
  auto.addEventListener("click", () => runMarkerAuto(layer, [channel], layer.displayAuto?.adjust || "both"));
  const shown = channel.label && channel.label !== channel.name ? channel.label + " (" + channel.name + ")" : channel.name;
  const name = el("span", { class: "cname", text: shown, title: "Click for the histogram, and to rename " + channel.name });
  name.addEventListener("click", () => {
    layer.histogramFor = layer.histogramFor === channel.index ? null : channel.index;
    showHistogram(layer);
    if (layer.histogramFor !== null && layer.histogramHolder) layer.histogramHolder.scrollIntoView({ block: "nearest" });
  });
  if (layer.histogramFor === channel.index) row.classList.add("picked");
  layer.inputs = layer.inputs || {};
  layer.inputs[channel.index] = { row: row, low: low, high: high };
  // Dye and exposure, as the image file records them, when those columns are switched on above the list.
  const meta = (layer.info.meta || [])[channel.index] || {};
  const columns = markerColumns();
  const extra = [];
  if (columns.dye) extra.push(el("span", { class: "cmeta", text: meta.dye || "–", title: "Dye (from the image file)" }));
  if (columns.exposure) extra.push(el("span", { class: "cmeta", text: meta.exposure_ms === null || meta.exposure_ms === undefined ? "–" : meta.exposure_ms + " ms", title: "Exposure (from the image file)" }));
  if (extra.length) row.classList.add("with-meta-" + extra.length);  // one or two more columns, so min, max and Auto stay
  row.append(check, color, name, ...extra, low, high, auto);
  // Studio's Segmentation step: N and M choose the nuclear marker and the cell border (membrane) markers.
  const roles = state.markerRoles;
  if (roles) {
    row.classList.add("with-roles");
    const toggle = (role, text, on, label) => {
      const button = el("button", { type: "button", class: "role-toggle " + role + (on ? " on" : ""), text,
        title: label + " " + channel.name, "aria-label": label + " " + channel.name, "aria-pressed": String(on) });
      button.addEventListener("click", () => roles.onChange(role, channel.name));
      return button;
    };
    row.insertBefore(toggle("membrane", "M", roles.membrane.includes(channel.name), "Cell border marker"), name);
    row.insertBefore(toggle("nuclear", "N", roles.nuclear === channel.name, "Nuclear marker"), row.querySelector(".role-toggle.membrane"));
  }
  const marker = linkedMarker(layer, channel);
  if (marker) {
    const link = el("input", { type: "checkbox", title: "Link this min to the raw RESOLVE threshold" });
    link.checked = channel.resolveLinked !== false;
    link.addEventListener("change", () => {
      channel.resolveLinked = link.checked;
      if (link.checked) setMarkerRange(layer, channel, marker.threshold, markerHigh(channel, marker.threshold));
      showHistogram(layer);
    });
    row.appendChild(el("label", { class: "resolve-link" }, [link, el("span", { text: "RESOLVE threshold" })]));
  }
  return row;
}

function select(options, value, onChange) {
  const node = el("select");
  options.forEach(([key, label]) => {
    const option = el("option", { value: key, text: label });
    if (key === value) option.selected = true;
    node.appendChild(option);
  });
  node.addEventListener("change", () => onChange(node.value));
  return node;
}

function cellControls(layer, body) {
  if(layer.selection){body.appendChild(el('p',{class:'hint',text:'Selected cells are highlighted in blue. Refine or label them in Studio’s Phenotyping step.'}));return;}
  body.appendChild(el("div", { class: "row" }, [el("label", { class: "lbl", text: "Draw" }),
    select([["outline", "Outlines"], ["fill", "Filled"], ["both", "Filled with outlines"]], layer.mode || "outline",
      (v) => { layer.mode = v; bump(layer); })]));
  const tables = layer.tables || [];
  if (tables.length) {
    body.appendChild(el("div", { class: "row" }, [el("label", { class: "lbl", text: "Table" }),
      select(tables.map((t) => [t.id, t.name]), layer.table, (v) => { layer.table = v; layer.color_by = ""; layer.columns = null; bump(layer); })]));
    const colourRow = el("div", { class: "row" }, [el("label", { class: "lbl", text: "Colour by" })]);
    body.appendChild(colourRow);
    columnsFor(layer).then((columns) => {
      const sorted = sortColumns(columns);
      if (layer.color_by && !sorted.includes(layer.color_by)) sorted.unshift(layer.color_by);
      colourRow.appendChild(select([["", "Nothing (single colour)"]].concat(sorted.map((c) => [c, c])), layer.color_by || "",
        (v) => { layer.color_by = v; layer.hidden = new Set(); if (v && layer.mode === "outline") layer.mode = "both"; bump(layer); }));
    }).catch((error) => colourRow.appendChild(el("span", { class: "hint", text: error.message })));
  }
  const outline = el("input", { type: "color", value: layer.outline || "#ffff00" });
  outline.addEventListener("change", () => { layer.outline = outline.value; bump(layer); });
  body.appendChild(el("div", { class: "row field" }, [el("label", { class: "lbl", text: "Outline" }), outline,
    help("Outline colour for 'Filled with outlines', and for outlines when no column is chosen.")]));
  if (layer.color_by && !layer.live) {
    const colourMap = el("div", { class: "row" }, [el("label", { class: "lbl", text: "Colour map" }),
      select([["viridis", "Viridis"], ["magma", "Magma"], ["inferno", "Inferno"], ["plasma", "Plasma"], ["cividis", "Cividis"], ["Reds", "Reds"], ["Blues", "Blues"]],
        layer.colormap || "viridis", (v) => { layer.colormap = v; bump(layer); })]);
    colourMap.hidden = true;  // shown once the legend says the column is numeric
    body.appendChild(colourMap);
    legendInto(layer, body, "/cells/" + layer.token + "/legend?t=" + encodeURIComponent(layer.table) +
      "&col=" + encodeURIComponent(layer.color_by) + "&cmap=" + encodeURIComponent(layer.colormap || "viridis"),
      (kind) => { colourMap.hidden = kind !== "numeric"; });
  }
  if (layer.resolve && !state.studio) liveResolveInto(layer, body);
}

// Which extra marker columns the image card shows (per browser): the dye and the exposure.
function markerColumns() {
  try { return JSON.parse(localStorage.getItem("spacemap-viewer-marker-columns") || "{}"); } catch (e) { return {}; }
}
function setMarkerColumn(name, on) {
  const columns = markerColumns(); columns[name] = on;
  try { localStorage.setItem("spacemap-viewer-marker-columns", JSON.stringify(columns)); } catch (e) { /* private mode */ }
}
