/* SPACEMAP viewer, part 10 of 12: live RESOLVE and Studio's Background, ROI and RESOLVE panels.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ------------------------------------------------------------ live RESOLVE
// Thresholds on raw marker values. Each change re-runs RESOLVE for this ROI on
// the server (the same phenotypes as Tab 7 for the same thresholds) and the
// cells are recoloured; counts per phenotype update as you drag.
// Opens live RESOLVE for this ROI: saved thresholds, else each marker's Auto background.
function startLiveButton(layer) {
  const start = el("button", { type: "button", class: "btn", text: "Start live RESOLVE" });
  start.addEventListener("click", async () => {
    start.disabled = true;
    start.textContent = "Reading this ROI's cells...";
    try {
      const live = await postJSON("/resolve/" + layer.resolve + "/open", {});
      const image = state.layers.find((l) => l.type === "image" && l.token === layer.image_token);
      const changes = {};
      if (image) {
        for (const marker of live.markers.filter((m) => m.source === "suggested")) {
          const channel = image.channels.find((c) => c.name === marker.marker);
          if (!channel) continue;
          start.textContent = "Estimating " + marker.marker + " background...";
          const result = await getJSON(autoQuery(image, channel, { method: image.autoMethod, scope: image.autoScope }));
          changes[marker.marker] = marker.threshold = result.threshold;
        }
      }
      if (Object.keys(changes).length) Object.assign(live, await postJSON("/resolve/" + layer.resolve + "/thresholds", { thresholds: changes }));
      layer.live = live;
      syncResolveRanges(layer);
      layer.liveHidden = new Set();
      if (layer.mode === "outline") layer.mode = "fill";
      bump(layer);
    } catch (error) {
      message("Live RESOLVE: " + error.message, true);
      start.disabled = false;
      start.textContent = "Start live RESOLVE";
    }
  });
  return start;
}

async function saveLiveThresholds(layer) {
  const live = layer.live;
  if (live.busy || live.pending) { message("Wait for the current threshold update before saving."); return; }
  try {
    const result = await postJSON("/resolve/" + layer.resolve + "/save", {});
    live.markers.forEach((m) => { m.source = "saved"; });
    message("Saved into " + result.saved + " for " + result.roi + ".");
    buildPanel();
  } catch (error) { message("Could not save: " + error.message, true); }
}

// Studio's Background step: the image's Auto controls and one level per marker.
// A level is the marker's minimum on the image, as in the image card.
function backgroundStudioPanel(holder, saved) {
  const image = state.layers.find((l) => l.type === "image" && l.inputs && !(l.info && l.info.rgb));
  holder.replaceChildren();
  if (!image) { holder.appendChild(el("p", { class: "muted", text: "Open an image to set its background levels." })); return; }
  holder.appendChild(autoControls(image));
  const stored = (saved && saved.levels) || null;
  const load = el("button", { type: "button", class: "btn small", text: "Load exported levels", hidden: true,
    title: "Set each marker's level to the value saved in " + ((saved && saved.table) || "Tab 2's table"),
    onclick: () => applyBackgroundLevels(stored) });
  const savedChip = chip("Not exported yet", "", saved && saved.table);
  // Compares the image's current levels with the saved table as levels change.
  const compare = () => {
    if (!stored) return;
    const differs = image.channels.some((c) => c.name in stored && Math.abs(Number(c.low) - stored[c.name]) > 0.05);
    savedChip.textContent = differs ? "Differs from exported" : "Same as exported";
    savedChip.className = "chip " + (differs ? "warn" : "ok");
    load.hidden = !differs;
  };
  // Before / after: the image as measured, or as removing the levels leaves it (the levels do not change).
  const view = (after) => { image.showBefore = !after; refreshLayer(image); before.classList.toggle("active", !after); afterButton.classList.toggle("active", after);
    before.setAttribute("aria-pressed", String(!after)); afterButton.setAttribute("aria-pressed", String(after)); };
  const before = el("button", { type: "button", class: "btn small", text: "Before", title: "The image as measured", onclick: () => view(false) });
  const afterButton = el("button", { type: "button", class: "btn small", text: "After", title: "What removing the levels leaves", onclick: () => view(true) });
  holder.appendChild(el("div", { class: "row studio-before-after" }, [el("span", { class: "lbl", text: "Show" }),
    el("div", { class: "segmented", role: "group", "aria-label": "Before or after background removal" }, [before, afterButton]),
    help("Before shows the markers as measured; After shows what removing the levels leaves. The levels are kept either way.")]));
  view(!image.showBefore);
  holder.appendChild(el("div", { class: "row section-title-row" }, [el("div", { class: "section-title", text: "Levels" }), savedChip, load,
    help("Background level per marker, in raw values. It is the marker's minimum on the image, so what you see is what " +
      "removing it leaves. Type a level, or press Auto for one marker; they are kept in the project as you go. " +
      "Export writes them as Tab 2's threshold table; the grey number is the level exported there.")]));
  // Studio's Background step: per marker, whether it goes into the new image, its colour and new name, its dye
  // and exposure, the blank cycle subtracted from it, its level and Auto (saved.markers; saved.onConfig keeps them).
  const config = saved && saved.markers ? saved.markers : null;
  const byName = {};
  (config ? config.markers : []).forEach((m) => { byName[m.name] = m; });
  const blanks = config ? config.blanks : [];
  const changed = () => {
    if (!config || !saved.onConfig) return;
    const keep = [], rename = {}, colors = {}, pairs = {};
    config.markers.forEach((m) => { if (m.keep) keep.push(m.name); if (m.rename) rename[m.name] = m.rename; if (m.new_color) colors[m.name] = m.new_color; if (m.subtract) pairs[m.name] = m.subtract; });
    saved.onConfig({ keep, rename, colors, blanks: pairs });
  };
  const list = el("div", { class: "background-levels" + (config ? " with-markers" : "") });
  if (config) {
    const all = el("input", { type: "checkbox", "aria-label": "Keep every marker", title: "Every marker into the new image (blanks too)" });
    all.checked = config.markers.every((m) => m.keep);
    all.addEventListener("change", () => { config.markers.forEach((m) => { m.keep = all.checked; }); list.querySelectorAll("input.keep").forEach((box) => { box.checked = all.checked; }); changed(); });
    list.appendChild(el("div", { class: "row level-row level-head" }, [all, el("span", { text: "" }), el("span", { class: "level-name", text: "Marker" }),
      el("span", { class: "level-meta", text: "Dye" }), el("span", { class: "level-meta", text: "Exp (ms)" }),
      blanks.length ? el("span", { class: "level-blank", text: "Blank" }) : null, el("span", { class: "level-saved-head", text: "Saved" }), el("span", { text: "Level" })]));
  }
  const rows = image.channels.map((channel) => {
    const input = el("input", { type: "number", step: "any", min: "0", class: "num", "aria-label": channel.name + " level",
      placeholder: "not set" });
    const auto = el("button", { type: "button", class: "btn small", text: "Auto", title: "Measure " + channel.name + "'s background",
      onclick: () => runMarkerAuto(image, [channel]) });
    input.addEventListener("change", () => {
      const value = Number(input.value);
      if (input.value === "" || !Number.isFinite(value) || value < 0) { sync(); return; }
      setMarkerRange(image, channel, value, markerHigh(channel, value));
    });
    const sync = () => {
      if (document.activeElement !== input) input.value = Number.isFinite(channel.low) ? Number(channel.low).toFixed(1) : "";
      auto.disabled = !!image.autoRun;
    };
    const kept = stored && channel.name in stored ? stored[channel.name] : null;
    const marker = byName[channel.name];
    const parts = [];
    if (marker) {
      const keep = el("input", { type: "checkbox", class: "keep", "aria-label": "Keep " + channel.name, title: "Into the background-removed image" });
      keep.checked = marker.keep;
      keep.addEventListener("change", () => { marker.keep = keep.checked; changed(); });
      const colour = el("input", { type: "color", value: marker.new_color || channel.color || marker.color, "aria-label": "Colour of " + channel.name });
      colour.addEventListener("change", () => {
        marker.new_color = colour.value; channel.color = colour.value; channel.custom = true;
        remember(image); refreshLayer(image); changed();
      });
      // The name itself, editable: a different name renames the marker in the new image (empty goes back to the original).
      const rename = el("input", { type: "text", value: marker.rename || channel.name, maxlength: "64", class: "level-rename",
        "aria-label": "Name of " + channel.name, title: "Type to rename " + channel.name + " in the background-removed image" });
      rename.addEventListener("change", () => {
        const value = rename.value.trim();
        marker.rename = value && value !== channel.name ? value : "";
        rename.value = marker.rename || channel.name; changed();
      });
      let blank = null;
      if (blanks.length) {
        if (marker.blank) blank = el("span", { class: "chip level-blank", text: "blank", title: "A blank cycle: subtracted from the markers that name it" });
        else {
          blank = el("select", { class: "level-blank", "aria-label": "Blank subtracted from " + channel.name });
          [["", "None"], ...blanks.map((b) => [b, b])].forEach(([value, text]) => blank.appendChild(el("option", { value, text })));
          blank.value = marker.subtract || "";
          blank.addEventListener("change", () => { marker.subtract = blank.value; changed(); });
        }
      }
      parts.push(keep, colour, el("span", { class: "level-name" }, [rename]),
        el("span", { class: "level-meta", text: marker.dye, title: "Dye" }), el("span", { class: "level-meta", text: marker.exposure, title: "Exposure (ms)" }), blank);
    } else {
      parts.push(el("span", { class: "swatch", style: "background:" + (channel.color || "#888") }), el("span", { class: "level-name", text: channel.name, title: channel.name }));
    }
    const levelParts = marker && marker.blank ? [el("span", { class: "level-saved" }), el("span", { class: "muted", text: "not levelled" })] :
      [el("span", { class: "level-saved", text: kept === null ? "" : Number(kept).toFixed(1), title: "Saved level" }), input, auto];
    list.appendChild(el("div", { class: "row level-row" + (marker && marker.blank ? " is-blank" : "") }, [...parts, ...levelParts]));
    sync();
    return sync;
  });
  holder.appendChild(list);
  image.syncLevels = () => { rows.forEach((sync) => sync()); compare(); };
  compare();
}

function applyBackgroundLevels(levels) {
  const image = state.layers.find((l) => l.type === "image" && l.inputs && !(l.info && l.info.rgb));
  if (!image || !levels) return;
  image.channels.forEach((channel) => {
    if (channel.name in levels) setMarkerRange(image, channel, levels[channel.name], markerHigh(channel, levels[channel.name]));
  });
  buildPanel();
}

// Studio's ROI step: the ROI layer's own list (names, import, Undo), when it is open.
function roisStudioPanel(holder) {
  holder.replaceChildren();
  const layer = state.layers.find((l) => l.type === "shapes" && l.id === "rois");
  if (!layer) return false;
  shapesControls(layer, holder);
  return true;
}

// The levels Studio saves: every marker with a level set.
function backgroundLevels() {
  const image = state.layers.find((l) => l.type === "image" && l.inputs && !(l.info && l.info.rgb));
  if (!image) return null;
  const levels = {};
  image.channels.forEach((channel) => { if (Number.isFinite(channel.low)) levels[channel.name] = Number(channel.low); });
  return { image: state.spec && state.spec.title, levels, method: image.autoMethod || "Triangle", inside: !!image.autoScope, parameters:thresholdOptions(image),
    missing: image.channels.filter((channel) => !Number.isFinite(channel.low)).map((channel) => channel.name) };
}

// Studio's RESOLVE method panel: the thresholds are the image markers' minimums,
// listed here compactly; each opens its cell histogram on request.
function resolveStudioPanel(holder) {
  const layer = state.layers.find((item) => item.type === "cells" && item.resolve);
  holder.replaceChildren();
  if (!layer) {
    const cells = state.layers.some((item) => item.type === "cells");
    holder.appendChild(el("div", { class: "row" }, [el("p", { class: "muted", text: cells ?
      "This project has no saved RESOLVE rules yet." : "Open an ROI with cells to use RESOLVE." }),
      cells ? help("Set up the phenotype rules once in classic Tab 7 (Resolve (Threshold)). Their marker thresholds are then set here, on the image.") : null]));
    return;
  }
  const box = el("div", { class: "live-resolve studio-live" });
  holder.appendChild(box);
  if (!layer.live) {
    box.appendChild(el("div", { class: "row" }, [startLiveButton(layer), help("Colours the cells by RESOLVE's phenotypes. " +
      "Each marker's threshold is its minimum on the image: move it here or in the image card and the phenotypes " +
      "update as Tab 7 would assign them. Starts from your saved threshold table; markers without one start from Auto.")]));
    return;
  }
  const live = layer.live;
  const cellsChip = chip("", "ok"), speed = chip("");
  const showInfo = () => {
    cellsChip.textContent = live.cells.toLocaleString() + " cells";
    cellsChip.title = live.left_out ? live.left_out + " cells left out: a marker has no value" : "";
    speed.textContent = live.ms + " ms";
    speed.title = "Time of the last update";
  };
  showInfo();
  const near = el("input", { type: "checkbox", "aria-label": "Highlight cells near a threshold" });
  near.checked = !!layer.nearThreshold;
  near.addEventListener("change", () => { layer.nearThreshold = near.checked; refreshLayer(layer); });
  const stop = el("button", { type: "button", class: "btn small", text: "Stop", title: "Colour the cells by a table column again",
    onclick: () => { layer.live = null; bump(layer); } });
  box.appendChild(el("div", { class: "row" }, [cellsChip, speed, el("label", { class: "inline muted" }, [near, el("span", { text: "Near threshold" })]),
    help("Highlights cells within 10% of a raw threshold."), stop]));
  box.appendChild(el("div", { class: "row section-title-row" }, [el("div", { class: "section-title", text: "Thresholds" }),
    help("Raw marker values. A threshold is the marker's minimum on the image, so changing it here or in the image card changes both. " +
      "Click a marker for its cell histogram and drag there. Click a cell to see why it got its phenotype.")]));
  const list = el("div", { class: "resolve-thresholds" });
  const rows = live.markers.map((m) => liveMarkerRow(layer, m, true));
  rows.forEach((row) => list.appendChild(row));
  box.appendChild(list);
  box.appendChild(el("div", { class: "row section-title-row" }, [el("div", { class: "section-title", text: "Phenotypes" }),
    help("Cells per phenotype in this ROI. Click a phenotype to hide or show it on the image.")]));
  const legend = el("div", { class: "legend", text: "Loading..." });
  box.appendChild(legend);
  live.refresh = () => { showInfo(); liveLegend(layer, legend); rows.forEach((row) => row.sync()); };
  liveLegend(layer, legend);
  const rules = el("details", { class: "studio-rules" });
  rules.appendChild(el("summary", { text: "Phenotype rules" }));
  rules.appendChild(phenotypePanelEditor(layer));
  box.appendChild(rules);
  box.appendChild(el("button", { type: "button", class: "btn", text: "Save thresholds for " + live.roi,
    title: "Writes this ROI's thresholds into " + (live.saved_as || "RESOLVE's threshold table") + ", the table Tab 7's RESOLVE reads. A backup is kept.",
    onclick: () => saveLiveThresholds(layer) }));
}

function liveResolveInto(layer, body) {
  const box = el("div", { class: "live-resolve" });
  body.appendChild(box);
  box.appendChild(el("div", { class: "row section-title-row" }, [el("div", { class: "section-title", text: "Live RESOLVE" }),
    help("Colour the cells by RESOLVE's phenotypes and change marker thresholds (raw values): " +
      "phenotypes update while you drag, as Tab 7 would assign them. Starts from your saved threshold table.")]));
  if (!layer.live) {
    box.appendChild(startLiveButton(layer));
    return;
  }
  const live = layer.live;
  const info = el("div", { class: "hint" });
  const showInfo = () => {
    info.textContent = live.cells.toLocaleString() + " cells" + (live.left_out ? " (" + live.left_out + " left out: a marker has no value)" : "") +
      " · last update " + live.ms + " ms";
  };
  showInfo();
  box.appendChild(info);
  const near = el("input", { type: "checkbox" });
  near.checked = !!layer.nearThreshold;
  near.addEventListener("change", () => { layer.nearThreshold = near.checked; refreshLayer(layer); });
  box.appendChild(el("label", { class: "row hint" }, [near, el("span", { text: "Highlight cells within 10% of a raw threshold" })]));
  box.appendChild(phenotypePanelEditor(layer));
  const legend = el("div", { class: "legend", text: "Loading..." });
  box.appendChild(legend);
  live.refresh = () => { showInfo(); liveLegend(layer, legend); };
  liveLegend(layer, legend);
  box.appendChild(el("div", { class: "hint", text: "Set thresholds in the image marker rows. Click a marker for its cell histogram, or unlink its min to adjust display contrast independently. Click a cell to see why it got its phenotype." }));
  // Markers absent from the image still need an editable threshold.
  const image = state.layers.find((l) => l.type === "image" && l.token === layer.image_token);
  live.markers.filter((m) => !image || !image.channels.some((c) => c.name === m.marker)).forEach((m) => box.appendChild(liveMarkerRow(layer, m)));
  const save = el("button", { type: "button", class: "btn", text: "Save thresholds for " + live.roi,
    title: "Writes this ROI's thresholds into " + (live.saved_as || "RESOLVE's threshold table") + ", the table Tab 7's RESOLVE reads. A backup is kept.",
    onclick: () => saveLiveThresholds(layer) });
  const stop = el("button", { type: "button", class: "btn", text: "Stop", title: "Colour the cells by a table column again" });
  stop.addEventListener("click", () => { layer.live = null; bump(layer); });
  box.appendChild(el("div", { class: "row" }, [save, stop]));
}

function phenotypePanelEditor(layer) {
  if (layer.panelEditor) return layer.panelEditor;
  const box = el("details", { class: "phenotype-editor" });
  layer.panelEditor = box;
  box.appendChild(el("summary", { text: "Edit phenotype panel" }));
  const body = el("div"); box.appendChild(body);
  let loaded = false, revision, panel;
  const draw = () => {
    body.textContent = "";
    body.appendChild(el("p", { class: "hint", text: "All markers within a group must be present. Any one group can match. Separate marker names with commas. If scores tie, the phenotype higher in the list wins. Saving updates the panel in Tab 7 and keeps a backup." }));
    panel.table.forEach((row, index) => {
      const item = el("div", { class: "panel-phenotype" });
      const name = el("input", { type: "text", value: row.Phenotype, "aria-label": "Phenotype name" });
      name.addEventListener("input", () => { row.Phenotype = name.value; });
      const move = (delta) => {
        const next = index + delta;
        if (next < 0 || next >= panel.table.length) return;
        [panel.table[index], panel.table[next]] = [panel.table[next], panel.table[index]]; draw();
      };
      item.appendChild(el("div", { class: "row" }, [name,
        el("button", { class: "btn small", text: "Up", onclick: () => move(-1) }),
        el("button", { class: "btn small", text: "Down", onclick: () => move(1) }),
        el("button", { class: "btn small", text: "Remove", onclick: () => { panel.table.splice(index, 1); draw(); } })]));
      panel.criteria_cols.concat(["Supportive", "Must not have"]).forEach((column) => {
        const markers = el("input", { type: "text", value: (row[column] || []).join(", "), "aria-label": row.Phenotype + " " + column });
        markers.addEventListener("input", () => { row[column] = markers.value.split(",").map((s) => s.trim()).filter(Boolean); });
        item.appendChild(el("div", { class: "row" }, [el("label", { class: "lbl", text: column }), markers]));
      });
      body.appendChild(item);
    });
    const status = el("div", { class: "hint", role: "status" });
    const save = el("button", { class: "btn", text: "Save panel", onclick: async () => {
      if (layer.live && (layer.live.busy || layer.live.pending)) { status.textContent = "Wait for the current threshold update before saving."; return; }
      save.disabled = true; status.textContent = "Saving and updating this ROI...";
      try {
        const result = await postJSON("/resolve/" + layer.resolve + "/panel", { panel: panel, revision: revision });
        layer.live = result.live; layer.panelEditor = null;
        layer.referencePhenotypes = result.live.phenotypes.slice();
        layer.referencePanelPromise = Promise.resolve(layer.referencePhenotypes);
        syncResolveRanges(layer); bump(layer);
        message("Panel saved. Backup: " + result.backup);
      } catch (error) { status.textContent = error.message; save.disabled = false; }
    } });
    body.appendChild(el("div", { class: "row wrap" }, [
      el("button", { class: "btn", text: "Add phenotype", onclick: () => { panel.table.push({ Phenotype: "New phenotype " + (panel.table.length + 1) }); draw(); } }),
      el("button", { class: "btn", text: "Add alternative group", onclick: () => {
        let i = 1; while (panel.criteria_cols.includes("Criteria " + i)) i++;
        panel.criteria_cols.push("Criteria " + i); draw();
      } }), save,
      el("button", { class: "btn", text: "Discard edits", onclick: () => { layer.panelEditor = null; buildPanel(); } })]));
    body.appendChild(status);
  };
  box.addEventListener("toggle", async () => {
    if (!box.open || loaded) return;
    loaded = true; body.textContent = "Reading the panel...";
    try {
      const response = await getJSON("/resolve/" + layer.resolve + "/panel");
      revision = response.revision; panel = response.panel; draw();
    } catch (error) { loaded = false; body.textContent = error.message; }
  });
  return box;
}

function liveLegend(layer, box) {
  const hide = Array.from(layer.liveHidden || []).join("\u001f");
  getJSON("/resolve/" + layer.resolve + "/legend?hide=" + encodeURIComponent(hide)).then((legend) => {
    box.textContent = "";
    (legend.items || []).forEach((item) => {
      const row = el("div", { class: "item" + (item.hidden ? " hidden" : ""), title: "Click to hide or show" }, [
        el("span", { class: "swatch", style: "background:" + item.color }),
        el("span", { text: item.value }),
        el("span", { class: "count", text: Number(item.count).toLocaleString() }),
      ]);
      row.addEventListener("click", () => {
        const hidden = layer.liveHidden || (layer.liveHidden = new Set());
        if (hidden.has(item.value)) hidden.delete(item.value); else hidden.add(item.value);
        refreshLayer(layer);
        liveLegend(layer, box);
      });
      box.appendChild(row);
    });
  }).catch((error) => { box.textContent = error.message; });
}

function sendLive(layer, changes) {
  const live = layer.live;
  if (!live) return;
  live.pending = Object.assign(live.pending || {}, changes);
  if (live.busy) return;  // the latest values go with the next request
  const run = async () => {
    const batch = live.pending;
    live.pending = null;
    live.busy = true;
    try {
      const result = await postJSON("/resolve/" + layer.resolve + "/thresholds", { thresholds: batch });
      if (layer.live === live) {
        Object.assign(live, { version: result.version, counts: result.counts, cells: result.cells, ms: result.ms });
        refreshLayer(layer);
        if (live.refresh) live.refresh();
        if (!live.pending) state.layers.filter((l) => l.type === "image" && l.token === layer.image_token)
          .forEach((image) => { if (image.refreshConfirmedHistogram) image.refreshConfirmedHistogram(); });
      }
    } catch (error) {
      message("Live RESOLVE: " + error.message, true);
      if (layer.live === live && !live.pending) {
        try {
          const current = await postJSON("/resolve/" + layer.resolve + "/open", {});
          Object.assign(live, current); syncResolveRanges(layer); buildPanel();
        } catch (reloadError) { message("Could not restore confirmed thresholds: " + reloadError.message, true); }
      }
    }
    live.busy = false;
    if (live.pending && layer.live === live) run();
  };
  run();
}

function liveMarkerRow(layer, m, compact = false) {
  const row = el("div", { class: "hist live-marker" + (compact ? " compact" : "") });
  const input = el("input", { type: "number", step: "any", class: "num", value: Number(m.threshold).toFixed(1),
    title: "Threshold on the raw marker value", "aria-label": m.marker + " threshold" });
  const source = el("span", { class: "hint" });
  const positive = el("div", { class: "hint" });
  const canvas = el("canvas", { width: 300, height: 60, title: "Drag to set " + m.marker + "'s threshold (raw value)" });
  // Compact rows (Studio) open the histogram on request, so a long panel reads no histograms up front.
  const name = compact ? el("button", { type: "button", class: "lbl marker-name", text: m.marker, title: "Show " + m.marker + "'s cell histogram",
    "aria-expanded": "false" }) : el("label", { class: "lbl", text: m.marker, title: m.marker });
  row.appendChild(el("div", { class: "row" }, [name, input, source]));
  row.appendChild(positive);
  row.appendChild(canvas);
  if (compact) { positive.hidden = true; canvas.hidden = true; }
  let hist = null, dragging = false;
  const labels = { saved: "saved", suggested: "suggested", changed: "changed" };
  const axis = () => [Math.asinh(hist.edges[0] / 5), Math.asinh(hist.edges[hist.edges.length - 1] / 5)];
  const draw = () => {
    source.textContent = labels[m.source] || m.source || "";
    if (!hist) return;
    const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height, n = hist.counts.length;
    const [a0, a1] = axis();
    const at = (Math.asinh(m.threshold / 5) - a0) / Math.max(a1 - a0, 1e-9) * w;
    const tx = Math.min(w - 1, Math.max(1, at));  // beyond the drawn range: shown at the edge
    const top = Math.max(...hist.counts.map((c) => Math.sqrt(c)), 1);
    let above = 0;
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < n; i++) {
      const x0 = i * w / n, width = w / n, height = Math.sqrt(hist.counts[i]) / top * (h - 2);
      const on = x0 + width / 2 > at;
      if (on) above += hist.counts[i];
      ctx.fillStyle = on ? "#4f7dff" : "#777";
      ctx.fillRect(x0, h - height, Math.max(width - 0.5, 0.5), height);
    }
    ctx.strokeStyle = "#ffd400";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(tx, 0); ctx.lineTo(tx, h); ctx.stroke();
    const total = hist.total || 1;
    const count = dragging || m.source === "changed" && hist.threshold !== m.threshold ? above : hist.positive;
    positive.textContent = (dragging ? "about " : "") + Number(count).toLocaleString() + " positive (" + (100 * count / total).toFixed(1) + "%)";
  };
  const load = () => getJSON("/resolve/" + layer.resolve + "/histogram?marker=" + encodeURIComponent(m.marker))
    .then((result) => { hist = result; draw(); }).catch((error) => { positive.textContent = error.message; });
  const valueAt = (clientX) => {
    const box = canvas.getBoundingClientRect(), [a0, a1] = axis();
    const f = Math.min(1, Math.max(0, (clientX - box.left) / box.width));
    return 5 * Math.sinh(a0 + f * (a1 - a0));
  };
  const set = (value) => {
    const image = state.layers.find((l) => l.type === "image" && l.token === layer.image_token);
    const channel = image && image.channels.find((c) => c.name === m.marker);
    if (channel && channel.resolveLinked !== false) setMarkerRange(image, channel, value, markerHigh(channel, value));
    else { m.threshold = value; m.source = "changed"; sendLive(layer, { [m.marker]: value }); }
    input.value = value.toFixed(1);
    draw();
  };
  canvas.addEventListener("pointerdown", (event) => {
    if (!hist) return;
    canvas.setPointerCapture(event.pointerId);
    dragging = true;
    set(valueAt(event.clientX));
  });
  canvas.addEventListener("pointermove", (event) => { if (dragging) set(valueAt(event.clientX)); });
  const end = () => { if (dragging) { dragging = false; load(); } };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  input.addEventListener("change", () => {
    const value = Number(input.value);
    if (Number.isFinite(value)) { set(value); if (!canvas.hidden) load(); }
  });
  if (compact) name.addEventListener("click", () => {
    const open = canvas.hidden;
    canvas.hidden = positive.hidden = !open;
    name.setAttribute("aria-expanded", String(open));
    if (open) load();
  });
  else load();
  draw();
  // A threshold changed elsewhere (the image card) shows here on the next update.
  row.sync = () => { if (document.activeElement !== input) input.value = Number(m.threshold).toFixed(1); draw(); };
  return row;
}

async function showWhy(layer, point) {
  const inspected = state.inspectedCell;
  try {
    const why = await getJSON("/resolve/" + layer.resolve + "/why?x=" + point.x + "&y=" + point.y);
    if (!inspected || state.inspectedCell !== inspected || inspected.result.label !== why.label) return;
    inspected.result.why = why;
    showCellCard(layer, inspected.result);
  } catch (error) { message("Why: " + error.message, true); }
}

function renderWhy(why, card) {
    if (!why.label) return;
    if (why.left_out) {
      card.appendChild(el("div", { class: "section", text: "Live RESOLVE: left out" }));
      card.appendChild(el("div", { class: "hint", text: why.reason || "" }));
      return;
    }
    card.appendChild(el("div", { class: "section", text: "Live RESOLVE: " + why.phenotype }));
    const values = {};
    (why.markers || []).forEach((m) => { values[m.marker] = m; });
    const describe = (marker) => {
      const m = values[marker];
      return m ? marker + " " + number(m.value) + (m.positive ? " > " : " <= ") + number(m.threshold) : marker + " (not measured)";
    };
    (why.phenotypes || []).forEach((p) => {
      const met = (p.criteria || []).filter((c) => c.met);
      let status = p.fired ? "criteria met" : "no criterion met";
      if (p.dropped_for_child) status = "met, but " + p.dropped_for_child + " is more specific";
      if (p.phenotype === why.phenotype) status = "assigned";
      const detail = p.fired
        ? met.map((c) => c.markers.join(" + ")).join("; ")
        : (p.criteria || []).map((c) => c.failing.map(describe).join(", ")).filter(Boolean).slice(0, 2).join(" | ");
      card.appendChild(el("div", { class: "why" + (p.phenotype === why.phenotype ? " on" : ""),
        title: "Tie score " + Number(p.strength || 0).toFixed(2) }, [
        el("b", { text: p.phenotype + ": " }), el("span", { text: status }),
        el("div", { class: "hint", text: detail }),
      ]));
    });
}
