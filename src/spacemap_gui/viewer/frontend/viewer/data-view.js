/* The data viewer: the Data tab of the panel below the image (beside Marker tiles), in Studio.
 * A cell table in pages: this image's rows (a merged table's image column), per-column filters, sorting, a column
 * picker, edits, copy and paste, export and save. It is linked to the image's cells both ways: a row outlines its
 * cell; a cell clicked on the image selects its row; selected cells can filter the rows; the rows' filter can be
 * shown on the image (the cell layer then draws only those cells). */
"use strict";

const dataView = { source: null, name: "", sources: [], rows: { this_image: true }, filters: [], sort: null, offset: 0, columns: null,
  selected: new Set(), anchor: null, follow: true, onlySelected: false, onImage: true, result: null, holder: null, busy: false };
const dataDirtySources = new Set();

async function dataShowMarkerReview(fetchSelection, alive) {
  if (dataDirtySources.size) throw Error('Save or discard your Data edits before showing reviewed cells.');
  const generation=state.initGeneration, sequence=state.groupPickSequence=(state.groupPickSequence||0)+1;
  const choice=dataView.choiceSequence;
  const current=()=>alive()&&generation===state.initGeneration&&sequence===state.groupPickSequence;
  const found=await fetchSelection();
  if(!current()||choice!==dataView.choiceSequence)return null;
  if(found.missing)return {missing:true,message:found.message};
  await state.phenotypingLink;
  if(!current()||choice!==dataView.choiceSequence)return null;
  if(found.table!==dataView.combined?.path)throw Error('This review belongs to another cell table.');
  const layer=state.layers.find(l=>l.type==='cells'&&l.combinedTable&&l.id===state.cellDataLayer)||state.layers.find(l=>l.type==='cells'&&l.combinedTable);
  if(!layer)throw Error('Open an image with cells linked to Phenotyping results.');
  const table=await cellTableWith(layer,found.colour_by?[found.colour_by]:[],layer.combinedTable,false);
  if(!current()||choice!==dataView.choiceSequence||!state.layers.includes(layer))return null;
  if(!table)throw Error('The reviewed cell table is unavailable for this image.');
  showDataTab();
  if(!dataView.source?.combined){
    const expectedSequence=sequence+(layer.table!==table?1:0),expectedChoice=(dataView.choiceSequence||0)+1;
    await dataChoose('combined');
    if(!alive()||generation!==state.initGeneration||expectedSequence!==state.groupPickSequence||
        expectedChoice!==dataView.choiceSequence||!dataView.source?.combined||!state.layers.includes(layer))return null;
  } else if(!current())return null;
  if(dataDirtySources.size)throw Error('Save or discard your Data edits before showing reviewed cells.');
  claimCellView(layer,found.name||'Marker review');
  layer.table=table;layer.filters=[];layer.hidden=new Set();layer.range=null;layer.live=null;
  if(found.colour_by){layer.color_by=found.colour_by;if((layer.mode||'outline')==='outline')layer.mode='both';}
  layer.flowSelection=found.selection.token;layer.flowPopulation=found.selection;layer.flowMode='filter';
  layer.visible=true;state.cellDataLayer=layer.id;
  dataView.flowSelection=found.selection.token;dataView.flowSelectionName=found.name||'Marker review';
  dataView.filters=[];dataView.groupPick=null;dataView.onlySelected=false;dataView.selected.clear();
  dataView.rows={this_image:true,column:dataView.info?.image_column,value:dataLayerRoi(layer)||''};
  bump(layer);buildPanel();dataDrawHead();
  const appliedSequence=state.groupPickSequence,appliedChoice=dataView.choiceSequence;
  await dataLoad();
  if(!alive()||generation!==state.initGeneration||appliedSequence!==state.groupPickSequence||appliedChoice!==dataView.choiceSequence)return null;
  return {count:found.selection.count,current:found.selection.by_image[dataLayerRoi(layer)]||0};
}
window.addEventListener('beforeunload', event => {
  if (dataDirtySources.size) { event.preventDefault(); event.returnValue = ''; }
});

function dataDocument() { return dataView.holder?.ownerDocument || document; }

function dataReadError(error) {
  if (!dataView.status) return;
  dataView.status.textContent = error.message;
  if (error.code === 'data_external_change') {
    dataView.queryReady = false;
    dataView.status.append(' ', el('button', {class:'btn small',text:'Reload saved data',
      title:'Discard the unsaved Data draft and open the latest saved table and result columns.',
      onclick:()=>dataHistory('discard')}));
  }
}

function dataRoute(action, query) {
  if(state.reviewer)return "/data/"+action+(query||"");
  const session = new URLSearchParams(location.search).get("session");
  return "/studio/" + encodeURIComponent(session) + "/data/" + action + (query || "");
}

function dataBody(extra) {
  const filters = dataView.filters.slice();
  if (dataView.onlySelected) {
    if (dataView.groupPick) filters.push({col:dataView.groupPick.column, values:dataView.groupPick.values});
    else {
      filters.push({ col: dataView.info?.label_column || "label", values: (dataView.pickedLabels || []).map(String) });
      const roiColumn = dataView.info?.roi_column || dataView.info?.image_column;
      if (dataView.pickedRoi && roiColumn) filters.push({col:roiColumn,values:[String(dataView.pickedRoi)]});
    }
  }
  return { source: dataView.source, rows: dataView.rows, filters, sort: dataView.sort, selection_column:dataView.groupPick?.column,
    ...(dataView.source?.combined && dataView.flowSelection ? {flow_selection:dataView.flowSelection} : {}), ...extra };
}

// The tables to look at: each cell layer's tables (linked to its cells), and tables opened here.
function dataSources() {
  const found = [];
  if (dataView.combined) found.push({key:'combined', text:'Phenotyping results', source:{combined:true}});
  state.layers.filter((l) => l.type === "cells").forEach((layer) => (layer.tables || []).forEach((t) =>
    found.push({ key: layer.id + "|" + t.id, text: cellDataInfo(layer,t.id).name + " · " + (layer.name || "Cells"), source: { cells: layer.token, table: t.id }, layer: layer.id })));
  (dataView.opened || []).forEach((item) => found.push({ key: "file|" + item.file, text: "Imported: " + item.name, source: { file: item.file }, path:item.path }));
  return found;
}

function dataLinkedLayer() {
  const key = dataView.sourceKey || "";
  if (key === 'combined') return state.layers.find(l=>l.type==='cells'&&l.combinedTable&&l.id===state.cellDataLayer)||state.layers.find(l => l.type === 'cells' && l.combinedTable);
  return state.layers.find((l) => l.type === "cells" && key.startsWith(l.id + "|"));
}

function dataLayerRoi(layer) {
  const table = dataView.source?.combined ? layer?.combinedTable : dataView.source?.cells === layer?.token ? dataView.source.table : layer?.table;
  return layer?.table_roles?.[table]?.roi || layer?.roi;
}

// Follow replacement measurements for the same cells without clearing the
// user's columns, filters, sorting, selected rows, or detached Data window.
async function dataRefreshTaggedTable(layer, previous) {
  const generation = state.initGeneration;
  if (!state.layers.includes(layer)) return;
  const own = dataView.source?.cells === layer.token && dataView.source.table === previous;
  const combinedImage = dataView.source?.combined && dataView.rows.this_image &&
    dataLinkedLayer() === layer && layer.combinedTable === previous;
  if (!own && !combinedImage) return;
  const key = layer.id + '|' + layer.table;
  const choice = dataView.choiceSequence = (dataView.choiceSequence || 0) + 1;
  dataView.loadSequence = (dataView.loadSequence || 0) + 1;
  clearTimeout(dataView.retry);
  dataView.source = {cells:layer.token, table:layer.table}; dataView.sourceKey = key;
  dataView.queryReady = false;
  const info = await postJSON(dataRoute('info'), dataBody({}));
  if (generation !== state.initGeneration || !state.layers.includes(layer) || choice !== dataView.choiceSequence || dataView.sourceKey !== key) return;
  if (dataPreparing(info, () => dataRefreshTaggedTable(layer, layer.table))) return;
  dataView.info = info;
  if (dataView.holder?.isConnected) { dataDrawHead(); await dataLoad(true); }
}

async function dataLoad(keepOffset) {
  if (!dataView.source || !dataView.holder) return;
  if (!keepOffset) dataView.offset = 0;
  dataView.status.textContent = "Reading…";
  dataView.loading = true;
  dataView.queryReady = false;
  document.querySelectorAll('.manual-reference-card').forEach(card=>card.refreshSelection?.());
  const sequence = dataView.loadSequence = (dataView.loadSequence || 0) + 1;
  try {
    const query = JSON.parse(JSON.stringify(dataBody({ offset: dataView.offset, columns: dataView.columns })));
    const found = await postJSON(dataRoute("query"), query);
    if (sequence !== dataView.loadSequence) return;
    if (dataPreparing(found, () => dataLoad(keepOffset))) return;
    dataView.result = found;
    if (found.edited) dataDirtySources.add(dataView.sourceKey); else dataDirtySources.delete(dataView.sourceKey);
    dataView.info={...dataView.info,renamed:found.renamed,removed:found.removed};
    dataSyncRemovedColumns(found.removed||[]);
    dataView.resultBody = query;
    dataView.queryReady = true;
    dataDraw();dataDrawHead();
  } catch (error) { if (sequence === dataView.loadSequence) dataReadError(error); }
  finally { if(sequence===dataView.loadSequence){dataView.loading=false;document.querySelectorAll('.manual-reference-card').forEach(card=>card.refreshSelection?.());} }
}

// Draft deletions stay undoable, but must not remain selectable on the linked cells.
function dataSyncRemovedColumns(removed) {
  const linked=dataLinkedLayer();let changed=false;
  for(const layer of state.layers.filter(l=>l.type==='cells'&&(l===linked||dataView.source?.combined&&l.combinedTable))){
    if(JSON.stringify(layer.stagedRemovedColumns||[])===JSON.stringify(removed))continue;
    layer.stagedRemovedColumns=removed;invalidateCellColumns(layer);changed=true;
    if(removed.includes(layer.color_by)){layer.color_by='';layer.hidden=new Set();layer.range=null;}
    layer.filters=(layer.filters||[]).filter(item=>!removed.includes(item.col));
    bump(layer);
  }
  if(changed)buildPanel();
}

// Pick a source: its info (the image column and its values), then the first page.
async function dataChoose(key) {
  const item = dataSources().find((s) => s.key === key);
  if (!item || dataView.busy) return;
  if(key!==dataView.sourceKey && dataDirtySources.has(dataView.sourceKey)){
    message('Save or discard your Data edits before switching tables.',true);return;
  }
  clearTimeout(dataView.retry);
  dataView.selectedColumns=new Set();dataView.undoViews=[];dataView.redoViews=[];
  const choiceSequence = dataView.choiceSequence = (dataView.choiceSequence || 0) + 1;
  dataView.loadSequence = (dataView.loadSequence || 0) + 1;
  Object.assign(dataView, { source: item.source, sourceKey: key, name:item.text, columns: null, filters: [], sort: null, selected: new Set(), offset: 0, result:null, info:null, queryReady:false, lastEditNotice:"", groupPick:null, onlySelected:false, flowSelection:null });
  dataDrawHead();dataView.status.textContent='Reading…';
  try {
    const info = await postJSON(dataRoute("info"), dataBody({}));
    if (choiceSequence !== dataView.choiceSequence) return;
    if (dataPreparing(info, () => {if(choiceSequence===dataView.choiceSequence)dataChoose(key);})) return;
    dataView.info = info;
    const linked = dataLinkedLayer();
    if(linked){const table=item.source.combined?linked.combinedTable:item.source.table;if(table&&linked.table!==table){claimCellView(linked,"Cell data changed");linked.table=table;linked.filters=[];linked.hidden=new Set();linked.range=null;linked.color_by="";invalidateCellColumns(linked);bump(linked);}state.cellDataLayer=linked.id;}
    dataView.awaitingImageScope = !!item.source.combined && !linked;
    dataView.rows = item.source.cells ? { this_image: true } : item.source.combined ?
      {this_image:!!linked, column:info.image_column, value:dataLayerRoi(linked) || info.values[0] || ''} :
      { this_image: !!info.image_column, column: info.image_column, value: info.values.includes(info.default) ? info.default :
        (info.values.find((v) => (state.spec?.title || "").includes(v)) || info.values[0] || "") };
  } catch (error) { if(choiceSequence===dataView.choiceSequence)dataReadError(error); return; }
  dataDrawHead(); dataLoad();
}

// A large CSV is read through its fast copy, made once in the background: say so and try again shortly.
function dataPreparing(found, again) {
  if (!found?.preparing) return false;
  dataView.status.textContent = found.message;
  clearTimeout(dataView.retry);
  dataView.retry = setTimeout(() => { if (dataView.holder?.isConnected) again(); }, 3000);
  return true;
}

// A filter changed: the rows again, and the image's cells with them when "Image follows" is on.
function dataFiltersChanged() {
  if (dataView.onImage && dataLinkedLayer()) dataShowOnImage(true);
  dataLoad();
}

function dataFilterLabel(item) {
  if (!item) return "All";
  if (item.values) return item.values.length === 1 ? item.values[0] : item.values.length + " values";
  if (item.contains !== undefined) return "has " + item.contains;
  const n = (v) => String(Math.round(v * 1000) / 1000);
  return item.lo !== null && item.hi !== null ? n(item.lo) + " – " + n(item.hi) : item.lo !== null ? "≥ " + n(item.lo) : "≤ " + n(item.hi);
}

// The top bar: source, rows, columns, the actions and the page.
function dataDrawHead() {
  const head = dataView.head;
  const sources = dataSources();
  const linkedSource=dataLinkedLayer(),activeName=linkedSource?cellDataInfo(linkedSource,dataView.source?.combined?linkedSource.combinedTable:dataView.source?.table).name:dataView.name||'No cell data available';
  const pick=el('span',{'aria-label':'Data source',class:'cell-data-current',text:activeName,title:dataView.info?.path||sources.find(s=>s.key===dataView.sourceKey)?.path||activeName});
  const chooseData=el('button',{class:'btn small',text:'Change cell data',onclick:()=>{document.querySelector('[role=tab][data-tab=layers]')?.click();const group=document.querySelector('[data-layer-group=cells]');if(group)group.open=true;const select=document.querySelector('[aria-label="Import cells"]');select?.scrollIntoView({block:'center'});select?.focus();}});
  const file = el("input", { type: "file", accept: ".csv,.tsv,.parquet", hidden: "" });
  file.addEventListener("change", async () => { if (file.files[0]) await dataOpenUpload(file.files[0]); file.value = ""; });
  const open = el("button", { class: "btn small", text: "Import…", title: "Add a CSV, TSV or Parquet to the source list. The current table stays open until you choose another.", onclick: () => file.click() });
  const parts = state.reviewer ? [el("strong",{text:"Cell data"}),pick,el("span",{class:"chip",text:"Read-only"})] : [el("strong",{text:"Cell data"}), pick, chooseData, open, file];
  if(dataView.source?.file)parts.push(el('button',{class:'btn small',text:'Link to Cells…',title:'Choose how this imported table matches the open image. Your current source stays selected.',onclick:()=>cellDataDialog({file_token:dataView.source.file,name:(dataView.opened||[]).find(o=>o.file===dataView.source.file)?.name||dataView.info?.name})}));
  if(dataView.source?.combined && dataView.combined?.name)parts.push(el("span",{class:"hint cell-data-file",text:dataView.combined.name}));
  if(dataView.flowSelection)parts.push(el("span",{class:"chip",text:dataView.flowSelectionName||"Flow gating selection"}),el("button",{class:"btn small",text:dataView.flowSelectionName?'Clear selection':'Clear Flow selection',onclick:()=>window.clearFlowGateView?.()}));
  const info = dataView.info;
  if (!state.reviewer && dataView.source && info && info.columns.length) {
    const only = el("input", { type: "checkbox", "aria-label": "This image only" }); only.checked = !!dataView.rows.this_image;
    const column = el("select", { "aria-label": "Image column" }, info.columns.map((c) => el("option", { value: c, text: c })));
    column.value = dataView.rows.column || info.image_column || "";
    const value = el("select", { "aria-label": "This image's value" }, info.values.map((v) => el("option", { value: v, text: v })));
    value.value = dataView.rows.value ?? info.default ?? info.values[0] ?? "";
    const update = async () => {
      dataView.awaitingImageScope = false;
      dataView.rows = { this_image: only.checked, column: column.value, value: value.value };
      if (column.value !== info.image_column) {
        try { dataView.info = await postJSON(dataRoute("info"), dataBody({})); } catch (_) { /* kept */ }
        dataDrawHead();
      }
      dataLoad();
    };
    [only, column, value].forEach((c) => c.addEventListener("change", update));
    parts.push(el("label", { class: "inline-check", title: "Only the rows of one image (a merged table's image column, such as batch_id)" }, [only, el("span", { text: "Image" })]), column, value);
  }
  const columns = el("button", { class: "btn small", text: "Columns", title: "Choose the columns shown", onclick: () => dataColumnPicker(columns) });
  const linked = dataLinkedLayer();
  const onImage = el("input", { type: "checkbox", "aria-label": "Image follows the filters" }); onImage.checked = dataView.onImage;
  onImage.disabled = !linked;
  onImage.addEventListener("change", () => {
    dataView.onImage = onImage.checked;
    if (onImage.checked) dataShowOnImage();
    else if (linked) { linked.filters = []; bump(linked); message("Data filters cleared. Other active cell selections are shown in Cells."); }
  });
  const show = el("label", { class: "inline-check", title: linked ? "Checked: hide image cells excluded by Data filters. Unchecked: show every cell." :
    "Open a cell layer's table to link it to the image" }, [onImage, el("span", { text: "Apply filters to viewer" })]);
  const follow = el("input", { type: "checkbox", "aria-label": "Follow the image" }); follow.checked = dataView.follow;
  follow.addEventListener("change", () => { dataView.follow = follow.checked; });
  const onlySel = el("input", { type: "checkbox", "aria-label": "Only selected cells" }); onlySel.checked = dataView.onlySelected;
  onlySel.addEventListener("change", () => { dataView.onlySelected = onlySel.checked; dataFiltersChanged(); });
  const exporting = el("select", { "aria-label": "Export" }, [el("option", { value: "", text: "Export…" }),
    el("option", { value: "shown", text: "These rows, shown columns (CSV)" }), el("option", { value: "all", text: "These rows, all columns (CSV)" })]);
  exporting.addEventListener("change", () => { const which = exporting.value; exporting.value = ""; if (which) dataExport(which === "shown"); });
  const allData=el('button',{class:'btn small',text:'Whole dataset','aria-label':'Show entire dataset',title:'Show all rows in the selected table, across images and batches. Clears filters; separate files are not combined.',onclick:()=>{
    dataView.awaitingImageScope=false;
    dataView.rows={...dataView.rows,this_image:false};dataView.filters=[];dataView.onlySelected=false;dataView.groupPick=null;dataView.follow=false;dataView.selected.clear();dataDrawHead();dataFiltersChanged();
  }});
  const currentImage=el('button',{class:'btn small',text:'Current image',disabled:linked?null:true,'aria-label':'Show current image data',
    title:'Show rows for the open image. Clear other filters.',onclick:()=>{
      dataView.awaitingImageScope=false;
      dataView.rows={this_image:true,column:info?.roi_column||info?.image_column||dataView.rows.column,value:dataLayerRoi(linked)};
      dataView.filters=[];dataView.onlySelected=false;dataView.groupPick=null;dataView.follow=true;dataView.selected.clear();dataDrawHead();dataFiltersChanged();
    }});
  const save=el('button',{class:'btn small',text:'Save',title:'Write all staged edits to this table; keep the original in previous/',onclick:()=>dataSave('overwrite')});
  save.disabled=dataView.busy||!dataView.result?.edited;
  const saveAs=el('button',{class:'btn small',text:'Save as…',onclick:()=>dataEditDialog({kind:'save_as'})});saveAs.disabled=dataView.busy||!dataView.source;
  const undo=el('button',{class:'btn small',text:'Undo',title:'Undo the last Data edit (Ctrl+Z)',onclick:()=>dataHistory('undo')});undo.disabled=dataView.busy||!dataView.result?.can_undo;
  const more=el('select',{'aria-label':'More data tools'},[el('option',{value:'',text:'Tools…'}),el('option',{value:'redo',text:'Redo'}),el('option',{value:'delete',text:'Delete rows…'}),el('option',{value:'discard',text:'Discard unsaved edits'})]);
  more.onchange=()=>{const action=more.value;more.value='';if(action==='delete')dataEditDialog({kind:'delete_rows',scope:dataView.selected.size?'selected':'filtered'});else if(action)dataHistory(action);};
  if(!state.reviewer)parts.push(currentImage,allData);
  parts.push(columns, show, el("label", { class: "inline-check", title: "Checked: clicking an image cell jumps to its Data row. Unchecked: keep your place in Data." }, [follow, el("span", { text: "Jump to clicked cell" })]),
    el("label", { class: "inline-check", title: "Only cells picked on the image or Data rows, or belonging to the clusters selected in the heatmap or cluster list. Rows of limits this to one ROI when checked." }, [onlySel, el("span", { text: "Picked cells only" })]),
    ...(state.reviewer?[]:[exporting, undo, save, saveAs, more]));
  if (dataView.source?.combined) parts.push(el('button',{class:'btn small',text:'Label cells…','aria-label':'Label Data cells',onclick:()=>{
    state.manualReferenceSource=dataView.selected.size?'data':'filtered';
    window.dispatchEvent(new CustomEvent('studio-reference-method',{detail:{method:'reference'}}));
  },title:'Assign a phenotype to selected rows, or all rows matching these filters, in your manual reference column.'}));
  head.replaceChildren(...parts);
  if(dataView.busy)head.querySelectorAll("button,select,input").forEach(c=>c.disabled=true);
}

function dataColumnPicker(anchor) {
  const result = dataView.result;
  if (!result) return;
  closeContextMenu();
  const shown = new Set(dataView.columns || result.shown);
  const filter = el("input", { type: "text", placeholder: "Find a column…", "aria-label": "Find a column" });
  const list = el("div", { class: "data-columns" });
  const draw = () => {
    const query = filter.value.toLowerCase();
    list.replaceChildren(...result.columns.filter((c) => c.toLowerCase().includes(query)).map((c) => {
      const tick = el("input", { type: "checkbox" }); tick.checked = shown.has(c);
      tick.addEventListener("change", () => { tick.checked ? shown.add(c) : shown.delete(c); });
      return el("label", { class: "inline-check" }, [tick, el("span", { text: c })]);
    }));
  };
  filter.addEventListener("input", draw); draw();
  const apply = el("button", { class: "btn small primary", text: "Show", onclick: () => {
    dataView.columns = result.columns.filter((c) => shown.has(c)); closeContextMenu(); dataLoad(true); } });
  const all = el("button", { class: "btn small", text: "All", onclick: () => { result.columns.forEach((c) => shown.add(c)); draw(); } });
  const none = el("button", { class: "btn small", text: "None", onclick: () => { shown.clear(); draw(); } });
  const menu = el("div", { id: "ctxmenu", class: "data-column-menu", role: "dialog" }, [filter, list, el("div", { class: "row" }, [all, none, apply])]);
  const box = anchor.getBoundingClientRect();
  menu.style.left = Math.max(8, Math.min(box.left, dataDocument().defaultView.innerWidth - 320)) + "px";
  dataDocument().body.append(menu);
  menu.style.top = Math.max(8, box.top - menu.offsetHeight - 4) + "px";
  filter.focus();
}

// The grid: a header with sorting, a filter box per column, then the page of rows.
function dataDraw() {
  const result = dataView.result;
  if (!result || !dataView.grid) return;
  const columns = result.shown;
  const filterOf = (c) => dataView.filters.find((f) => f.col === c);
  const head = el("tr", {}, [el("th", { class: "data-rowno", text: "#" }), ...columns.map((c) => {
    const sorted = dataView.sort?.col === c;
    return el("th", {class:dataView.selectedColumns?.has(c)?'column-selected':'', 'data-column':c,tabindex:0,
      title:c+': click to sort and select; right-click for editing tools',oncontextmenu:event=>dataEditorMenu(event,c),ondblclick:()=>dataEditDialog({kind:'rename_column',column:c}),onkeydown:event=>{if(event.key==='F2'){event.preventDefault();dataEditDialog({kind:'rename_column',column:c});}},onclick: event => {
      if(event.ctrlKey||event.metaKey){(dataView.selectedColumns||=new Set()).has(c)?dataView.selectedColumns.delete(c):dataView.selectedColumns.add(c);dataDraw();return;}
      if(event.shiftKey&&dataView.columnAnchor){const a=columns.indexOf(dataView.columnAnchor),b=columns.indexOf(c);dataView.selectedColumns=new Set(columns.slice(Math.min(a,b),Math.max(a,b)+1));dataDraw();return;}
      dataView.selectedColumns=new Set([c]);dataView.columnAnchor=c;
      dataView.sort = sorted && !dataView.sort.desc ? { col: c, desc: true } : sorted ? null : { col: c, desc: false }; dataLoad(); } },
      [el("span", { text: c }), sorted ? el("span", { class: "muted", text: dataView.sort.desc ? " ▼" : " ▲" }) : null].filter(Boolean));
  })]);
  const filters = el("tr", { class: "data-filters" }, [el("th"), ...columns.map((c) => {
    const item = filterOf(c);
    const button = el("button", { type: "button", class: "data-filter-btn" + (item ? " on" : ""), "aria-label": "Filter " + c,
      text: (result.kinds[c] === "value" ? "▾ " : "☰ ") + dataFilterLabel(item),
      title: (result.kinds[c] === "value" ? "Numbers: choose a range on the histogram or type From / To." : "Groups: tick the values to keep.") + " Click to change." });
    button.addEventListener("click", (event) => { event.stopPropagation(); dataFilterMenu(button, c, result.kinds[c]); });
    return el("th", {}, [button]);
  })]);
  const body = result.rows.map((row, index) => {
    const [id, ...values] = row;
    const identity = result.identity?.[id];
    const picked = dataView.groupPick ? dataView.groupPick.values.includes(String(identity?.group)) :
      dataView.selected.has(id) || (identity?.roi === dataView.pickedRoi && (dataView.pickedLabels || []).includes(Number(identity?.label)));
    const tr = el("tr", { class: picked ? "selected" : "", "data-row": String(id) },
      [el("td", { class: "data-rowno", text: String(result.offset + index + 1) }), ...values.map((value, column) => {
        const td = el("td", { class:dataView.selectedColumns?.has(columns[column])?"column-selected":"", text: value === null ? "" : typeof value === "number" ? String(Math.round(value * 1e6) / 1e6) : String(value) });
        td.addEventListener("dblclick", () => dataEditCell(td, id, columns[column], value));
        td.dataset.col = columns[column];
        return td;
      })]);
    tr.addEventListener("click", (event) => {if(!event.target.closest("input"))dataSelectRow(event, id, row);});
    tr.addEventListener("contextmenu",event=>dataEditorMenu(event,event.target.closest("td")?.dataset.col,id));
    return tr;
  });
  dataView.grid.replaceChildren(el("table", { class: "data-grid" }, [el("thead", {}, [head, filters]), el("tbody", {}, body)]));
  const from = result.matched ? result.offset + 1 : 0, to = Math.min(result.offset + result.rows.length, result.matched);
  dataView.status.replaceChildren(
    el("button", { class: "btn tiny", text: "◀", disabled: result.offset ? null : "", onclick: () => { dataView.offset = Math.max(0, result.offset - 200); dataLoad(true); } }),
    el("span", { text: " " + from.toLocaleString() + "–" + to.toLocaleString() + " of " + result.matched.toLocaleString() +
      (result.matched !== result.total ? " (" + result.total.toLocaleString() + " rows)" : " rows") + " " }),
    el("button", { class: "btn tiny", text: "▶", disabled: to < result.matched ? null : "", onclick: () => { dataView.offset = result.offset + 200; dataLoad(true); } }),
    el("input",{type:"number",min:1,max:Math.max(1,Math.ceil(result.matched/200)),value:Math.floor(result.offset/200)+1,class:"data-page-number","aria-label":"Data page",title:"Go to page",onchange:event=>{dataView.offset=(Math.max(1,Number(event.target.value)||1)-1)*200;dataLoad(true);}}),
    dataView.filters.length ? el("button", { class: "btn tiny", text: "Clear filters", onclick: () => { dataView.filters = []; dataFiltersChanged(); } }) : null,
    dataView.lastEditNotice ? el("span",{class:"muted",text:dataView.lastEditNotice}) : null,
    result.edited ? el("span", { class: "chip warn", text: result.edited + " edits not saved" }) : null,
    el("span", { class: "muted data-selection-count", text: dataView.selected.size ? " · " + dataView.selected.size + " selected (Ctrl+C copies)" : "" }));
}

// Click selects a row (Ctrl adds, Shift a range); a row with a cell number outlines it on the image.
function dataSelectRow(event, id, row) {
  dataView.groupPick = null;
  dataView.pickedFromViewer = false;
  const ids = dataView.result.rows.map((r) => r[0]);
  if (event.shiftKey && dataView.anchor !== null) {
    const a = ids.indexOf(dataView.anchor), b = ids.indexOf(id);
    if (a >= 0 && b >= 0) ids.slice(Math.min(a, b), Math.max(a, b) + 1).forEach((r) => dataView.selected.add(r));
  } else if (event.ctrlKey || event.metaKey) {
    dataView.selected.has(id) ? dataView.selected.delete(id) : dataView.selected.add(id);
  } else dataView.selected = new Set([id]);
  dataView.anchor = id;
  state.manualReferenceSource = 'data';
  for(const tr of dataView.grid.querySelectorAll('tbody tr'))tr.classList.toggle('selected',dataView.selected.has(Number(tr.dataset.row)));
  const count=dataView.status.querySelector('.data-selection-count');if(count)count.textContent=dataView.selected.size?' · '+dataView.selected.size+' selected (Ctrl+C copies)':'';
  dataView.grid.focus({preventScroll:true});
  document.querySelectorAll('.manual-reference-card').forEach(card=>card.refreshSelection?.());
  const layer = dataLinkedLayer();
  const shown = dataView.result.shown;
  if (!layer) return;
  const rows = dataView.result.rows.filter((r) => dataView.selected.has(r[0]));
  const at = (r, names) => { const i = names.map((n) => shown.indexOf(n)).find((k) => k >= 0); return i === undefined || i < 0 ? null : r[i + 1]; };
  const identities = rows.map(r => dataView.result.identity?.[r[0]]);
  const labels = identities.filter(identity => identity && String(identity.roi) === String(dataLayerRoi(layer))).map(identity => Number(identity.label)).filter(v => Number.isInteger(v) && v > 0);
  document.dispatchEvent(new CustomEvent('spacemap-cells-picked', {detail:{layer,labels,fromData:true}}));
  if (!labels.length) { clearCellInspection({notify:false});if (rows.length) message("Open this row’s ROI image to see its cells."); return; }
  state.cardSeq += 1;
  if (labels.length === 1) {
    const pickedRow = rows.find(r => String(dataView.result.identity?.[r[0]]?.roi) === String(dataLayerRoi(layer)));
    const x = at(pickedRow, ["x_centroid", "centroid_x", "X"]), y = at(pickedRow, ["y_centroid", "centroid_y", "Y"]);
    const point = x !== null && y !== null ? { x, y } : null;
    if (point && state.viewer) {
      const item = baseItem();
      if (item) state.viewer.viewport.panTo(item.imageToViewportCoordinates(x, y));
    }
    outlineCell(layer, labels[0], point);
  } else outlineSelectedCells(layer, labels.map((label) => ({ label })), state.cardSeq);
}

function dataEditCell(td, id, column, value) {
  if(state.reviewer)return;
  const input = el("input", { type: "text", value: value === null ? "" : String(value), class: "data-edit", "aria-label": "Edit " + column });
  td.replaceChildren(input); input.focus(); input.select();
  let settled=false;
  input.addEventListener('click',event=>event.stopPropagation());
  const done=async keep=>{
    if(settled||!input.isConnected)return;settled=true;
    if(keep&&input.value!==(value===null?'':String(value)))await dataMutate({kind:'cells',changes:[{row:id,col:column,value:input.value}]});
    else dataDraw();
  };
  input.addEventListener("keydown", (event) => { if (event.key === "Enter") done(true); if (event.key === "Escape") done(false); event.stopPropagation(); });
  input.addEventListener("blur", () => done(true));
}

// Ctrl+C: the selected rows (shown columns) as tab-separated text. Ctrl+V: pasted values down one column from the
// selected row (one value per line).
function dataKeys(event) {
  if(state.reviewer && event.key.toLowerCase()!=="c")return;
  if (!dataView.result || !(event.ctrlKey || event.metaKey) || event.target.matches("input, select, textarea")) return;
  if(event.key.toLowerCase()==='z'){event.preventDefault();dataHistory(event.shiftKey?'redo':'undo');return;}
  if(event.key.toLowerCase()==='y'){event.preventDefault();dataHistory('redo');return;}
  if(event.key.toLowerCase()==='s'){event.preventDefault();dataSave('overwrite');return;}
  const shown = dataView.result.shown;
  if (event.key === "c" && dataView.selected.size) {
    const rows = dataView.result.rows.filter((r) => dataView.selected.has(r[0]));
    const text = [shown.join("\t"), ...rows.map((r) => r.slice(1).map((v) => v === null ? "" : String(v)).join("\t"))].join("\n");
    dataDocument().defaultView.navigator.clipboard?.writeText(text).then(() => message(rows.length + " rows copied.")).catch(() => message("The browser did not allow copying.", true));
    event.preventDefault();
  }
  if (event.key === "v" && dataView.selected.size) {
    event.preventDefault();
    dataDocument().defaultView.navigator.clipboard?.readText().then(async (text) => {
      const column = window.prompt("Paste into which column?", dataView.pasteColumn || shown[shown.length - 1]);
      if (!column || !shown.includes(column)) return;
      dataView.pasteColumn = column;
      const lines = text.replace(/\r/g, "").split("\n").filter((l, i, all) => l !== "" || i < all.length - 1);
      const ids = dataView.result.rows.map((r) => r[0]);
      const start = ids.indexOf(Math.min(...Array.from(dataView.selected).map((id) => ids.indexOf(id) >= 0 ? id : Infinity)));
      const changes = lines.map((value, i) => ids[start + i] === undefined ? null : { row: ids[start + i], col: column, value: value.split("\t")[0] }).filter(Boolean);
      try { await dataMutate({kind:"cells",changes}); }
      catch (error) { message(error.message, true); }
      dataLoad(true);
    }).catch(() => message("The browser did not allow pasting.", true));
  }
}

async function dataShowOnImage(quiet) {
  const layer = dataLinkedLayer();
  if (!layer) return;
  // An empty selection leaves the image available for picking cells, while Data can show zero selected rows.
  const hasSelection = dataView.groupPick?.values.length || dataView.pickedLabels?.length;
  const filters = dataView.onlySelected && !hasSelection ? dataView.filters.slice() : dataBody().filters;
  if (dataView.source?.combined && dataView.rows.this_image && dataView.rows.column) filters.push({col:dataView.rows.column,values:[String(dataView.rows.value)]});
  claimCellView(layer,dataView.groupPick ? "Phenotyping clusters" : "Data filters");
  if(!dataView.groupPick){layer.hidden=new Set();layer.range=null;}
  layer.filters = filters;
  const table = dataView.sourceKey === 'combined' ? layer.combinedTable : (dataView.sourceKey || "").split("|")[1];
  if (table && layer.table !== table) { layer.table = table; layer.columns = null; layer.color_by = ""; }
  if (!layer.visible) setVisible(layer, true);
  bump(layer);
  if (!quiet) message(filters.length ? "The image shows only the cells of these rows (Cells > Filters; × there shows all again)." : "Data filters cleared. Other active cell selections are shown in Cells.");
}

function dataRenameSelection() {
  // Viewer selections may span more than the current page; local mask IDs need their ROI.
  const layer = dataLinkedLayer();
  if (dataView.pickedFromViewer && layer && dataView.pickedLayer === layer.token && dataView.pickedLabels?.length) {
    const filters = [{col:dataView.info?.label_column || 'label', values:dataView.pickedLabels.map(String)}];
    const roi = dataView.info?.roi_column || dataView.info?.image_column;
    if (roi && dataView.pickedRoi != null) filters.push({col:roi, values:[String(dataView.pickedRoi)]});
    else if (dataView.source?.combined) return {count:0};
    return {count:dataView.pickedLabels.length, filters};
  }
  return {count:dataView.selected.size};
}

// A column's filter pop-up. Groups (text): every value with its count, ticked when kept; a search narrows the list
// and All / None act on the values listed. Numbers: a histogram (drag across it to choose a range) and From / To.
// Counts are over the rows the other filters keep. Every change applies at once.
async function dataFilterMenu(anchor, column, kind) {
  closeContextMenu();
  const sourceKey = dataView.sourceKey;
  const menu = el("div", { id: "ctxmenu", class: "data-column-menu data-filter-menu", role: "dialog", "aria-label": "Filter " + column },
    [el("strong", { text: column }), el("div", { class: "muted", text: "Counting…" })]);
  const box = anchor.getBoundingClientRect();
  menu.style.left = Math.max(8, Math.min(box.left, dataDocument().defaultView.innerWidth - 330)) + "px";
  dataDocument().body.append(menu);
  const place = () => { menu.style.top = Math.max(8, box.top - menu.offsetHeight - 4) + "px"; };
  place();
  const body = dataBody({ column, limit: 2000, bins: 40 });
  body.filters = body.filters.filter((f) => f.col !== column);
  let found;
  try { found = await postJSON(dataRoute("summary"), body); }
  catch (error) { menu.lastChild.textContent = error.message; return; }
  if (!menu.isConnected) return;
  let timer = null, pendingFilter = null;
  const apply = (item, now) => {
    clearTimeout(timer);
    const run = () => { pendingFilter = null; if (sourceKey !== dataView.sourceKey) return;
      dataView.filters = dataView.filters.filter((f) => f.col !== column).concat(item ? [item] : []); dataFiltersChanged(); };
    pendingFilter = run;
    if (now) run(); else timer = setTimeout(run, 250);
  };
  const current = dataView.filters.find((f) => f.col === column);
  const n = (v) => (Math.abs(v) >= 1000 || Number.isInteger(v) ? Math.round(v) : Math.round(v * 1000) / 1000).toLocaleString();
  const clear = el("button", { class: "btn small", text: "Clear", title: "Keep every value of " + column, onclick: () => { apply(null, true); closeContextMenu(); } });
  const close = el("button", { class: "btn small primary", text: "Done", onclick: () => closeContextMenu() });
  let content;
  if (found.kind === "group") {
    let kept = new Set(current?.values || found.groups.map((g) => g[0]));
    let renaming = false, activeEditor = null;
    const search = el("input", { type: "text", placeholder: "Find a value…", "aria-label": "Find a value" });
    const list = el("div", { class: "data-columns data-values" });
    const selectedOnly = el('input', {type:'checkbox', 'aria-label':'Rename selected cells only'});
    selectedOnly.disabled = !dataRenameSelection().count;
    const scope = el('label', {class:'inline-check data-rename-scope',
      title:'Unchecked: rename every matching value in the entire dataset, including other images. Checked: only selected cells. Changes remain undoable until Save.'},
      [selectedOnly, el('span', {text:'Selected cells only'})]);
    const status = el('div', {class:'muted data-rename-status', role:'status'});
    const empty = el('div', {class:'muted'});
    const updateEmpty = () => { empty.hidden = !found.empty; empty.textContent = found.empty.toLocaleString() + ' rows have no value.'; };
    const shownValues = () => found.groups.filter((g) => g[0].toLowerCase().includes(search.value.trim().toLowerCase()));
    const changed = () => {
      const all = found.groups.every((g) => kept.has(g[0])) && found.distinct <= found.groups.length;
      apply(all ? null : { col: column, values: Array.from(kept) });
      count.textContent = kept.size.toLocaleString() + " of " + found.distinct.toLocaleString() + " values kept";
    };
    const draw = () => list.replaceChildren(...shownValues().map(([value, number]) => {
      const tick = el("input", { type: "checkbox", "aria-label": value }); tick.checked = kept.has(value);
      tick.addEventListener("change", () => { tick.checked ? kept.add(value) : kept.delete(value); changed(); });
      const name = el('input', {type:'text', class:'data-value-name', value, 'aria-label':'Rename value '+value,
        title:'Enter or leave this box to rename. Escape cancels. Applies to the entire dataset unless Selected cells only is checked.'});
      const commit = async () => {
        const next = name.value;
        if (renaming || next === value || sourceKey !== dataView.sourceKey) return;
        if (!next.trim()) { status.textContent = 'Enter a name; use Clear selected values to remove a label.'; name.value = value; return; }
        clearTimeout(timer); pendingFilter?.();
        const selected = selectedOnly.checked;
        const target = dataRenameSelection();
        if (selected && !target.count) { status.textContent = 'Select cells first.'; return; }
        renaming = true; status.textContent = 'Renaming…';
        menu.querySelectorAll('input,button').forEach(input => { input.disabled = true; });
        try {
          const ok = await dataMutate({kind:'replace', column, old:value, value:next, scope:selected ? 'selected' : 'all'},
            selected && target.filters ? {rows:{this_image:false}, filters:target.filters} : null);
          if (!ok) { name.value = value; status.textContent = dataView.lastError || 'Could not rename this value.'; return; }
          activeEditor = null;
          if (!menu.isConnected || sourceKey !== dataView.sourceKey) return;
          const query = dataBody({column, limit:2000, bins:40});
          query.filters = query.filters.filter(f => f.col !== column);
          found = await postJSON(dataRoute('summary'), query);
          if (!menu.isConnected || sourceKey !== dataView.sourceKey) return;
          kept = new Set(dataView.filters.find(f => f.col === column)?.values || found.groups.map(g => g[0]));
          search.value = ''; draw(); updateEmpty();
          count.textContent = kept.size.toLocaleString() + ' of ' + found.distinct.toLocaleString() + ' values kept';
          status.textContent = dataView.lastAffected ? dataView.lastAffected.toLocaleString() + ' rows renamed. Save to keep.' :
            (selected ? 'No matching selected cells.' : 'No matching values.');
          place();
        } catch (error) { status.textContent = error.message; }
        finally {
          renaming = false;
          menu.querySelectorAll('input,button').forEach(input => { input.disabled = false; });
          selectedOnly.disabled = !dataRenameSelection().count;
        }
      };
      name.addEventListener('focus', () => { activeEditor = commit; });
      name.addEventListener('input', () => { activeEditor = commit; });
      // Let a click on the scope checkbox finish before applying an edit on blur.
      name.addEventListener('blur', () => setTimeout(commit, 0));
      name.addEventListener('keydown', event => {
        event.stopPropagation();
        if (event.key === 'Enter') { event.preventDefault(); commit(); }
        if (event.key === 'Escape') { event.preventDefault(); name.value = value; activeEditor = null; name.blur(); }
      });
      return el('div', {class:'data-value'}, [tick, name, el('span', {class:'muted data-value-count', text:number.toLocaleString()})]);
    }));
    menu.beforeClose = () => { if (!renaming) activeEditor?.(); };
    search.addEventListener("input", draw);
    const count = el("span", { class: "muted", text: kept.size.toLocaleString() + " of " + found.distinct.toLocaleString() + " values kept" });
    const every = el("button", { class: "btn small", text: "All", title: "Tick the values listed", onclick: () => { shownValues().forEach((g) => kept.add(g[0])); draw(); changed(); } });
    const none = el("button", { class: "btn small", text: "None", title: "Untick the values listed", onclick: () => { shownValues().forEach((g) => kept.delete(g[0])); draw(); changed(); } });
    draw(); updateEmpty();
    content = [search, el("div", { class: "row data-value-tools" }, [every, none, count]), scope, list, status,
      found.distinct > found.groups.length ? el("div", { class: "muted", text: "The " + found.groups.length.toLocaleString() + " most common values are listed." }) : null,
      empty];
  } else if (found.bins) {
    const from = el("input", { type: "number", step: "any", "aria-label": "From", placeholder: n(found.min), value: current?.lo ?? "" });
    const to = el("input", { type: "number", step: "any", "aria-label": "To", placeholder: n(found.max), value: current?.hi ?? "" });
    const peak = Math.max(...found.bins, 1);
    const bars = found.bins.map((value, i) => el("div", { class: "data-bar", "data-bin": String(i),
      title: n(found.edges[i]) + " – " + n(found.edges[i + 1]) + ": " + value.toLocaleString() + " rows" },
      [el("span", { style: "height:" + Math.max(value ? 4 : 0, Math.sqrt(value / peak) * 100) + "%" })]));
    const histogram = el("div", { class: "data-histogram", title: "Drag across the bars to keep that range" }, bars);
    const mark = () => {
      const lo = from.value === "" ? -Infinity : Number(from.value), hi = to.value === "" ? Infinity : Number(to.value);
      bars.forEach((bar, i) => bar.classList.toggle("in", found.edges[i + 1] >= lo && found.edges[i] <= hi));
    };
    const changed = (now) => {
      mark();
      const lo = from.value === "" ? null : Number(from.value), hi = to.value === "" ? null : Number(to.value);
      apply(lo === null && hi === null ? null : { col: column, lo, hi }, now);
    };
    [from, to].forEach((input) => input.addEventListener("input", () => changed(false)));
    let start = null;
    const binOf = (event) => { const bar = event.target.closest(".data-bar"); return bar ? Number(bar.dataset.bin) : null; };
    const pick = (a, b) => { const i = Math.min(a, b), j = Math.max(a, b); from.value = String(found.edges[i]); to.value = String(found.edges[j + 1]); mark(); };
    histogram.addEventListener("pointerdown", (event) => { start = binOf(event); if (start !== null) { histogram.setPointerCapture(event.pointerId); pick(start, start); } });
    histogram.addEventListener("pointermove", (event) => {
      if (start === null) return;
      const rect = histogram.getBoundingClientRect();
      const i = Math.max(0, Math.min(bars.length - 1, Math.floor((event.clientX - rect.left) / rect.width * bars.length)));
      pick(start, i);
    });
    histogram.addEventListener("pointerup", () => { if (start !== null) { start = null; changed(true); } });
    mark();
    content = [histogram, el("div", { class: "data-axis muted" }, [el("span", { text: n(found.min) }), el("span", { text: n(found.max) })]),
      el("div", { class: "row" }, [el("span", { text: "From" }), from, el("span", { text: "to" }), to]),
      el("div", { class: "muted", text: "Median " + n(found.median) + " · mean " + n(found.mean) + " · " + found.count.toLocaleString() + " rows" +
        (found.empty ? " (" + found.empty.toLocaleString() + " empty)" : "") })];
  } else content = [el("div", { class: "muted", text: "No numbers in these rows." })];
  menu.replaceChildren(el("strong", { text: column }), ...content.filter(Boolean), el("div", { class: "row" }, [clear, el("span", { class: "grow" }), close]));
  place();
  menu.querySelector("input")?.focus();
}

function dataExport(shownOnly) {
  const query = encodeURIComponent(JSON.stringify(dataBody({ columns: shownOnly ? dataView.result?.shown : null })));
  const link = el("a", { href: api(dataRoute("export", "?q=" + query)) });
  dataDocument().body.appendChild(link); link.click(); link.remove();
}

async function dataSave(mode,name) {
  if(state.reviewer)return;
  if(dataView.busy||!dataView.source)return false;
  dataView.busy=true;dataView.lastError='';dataView.holder.classList.add('data-busy');dataDrawHead();
  const request=dataBody({mode,name,revision:dataView.result?.revision});
  dataView.status.textContent='Saving…';
  const poll=setInterval(async()=>{try{const status=await postJSON(dataRoute('status'),request);if(status.progress?.state==='saving')dataView.status.textContent='Saving… '+Math.round(100*status.progress.done/Math.max(1,status.progress.total))+'%';}catch(_){}},1000);
  try{
    const saved=await postJSON(dataRoute('save'),request);
    dataDirtySources.delete(dataView.sourceKey);
    dataView.undoViews=[];dataView.redoViews=[];dataView.selected.clear();dataView.lastEditNotice='';
    if(mode==='overwrite'){
      const linked=dataLinkedLayer(),renamed=dataView.result?.renamed||{},removed=new Set(dataView.result?.removed||[]);
      for(const layer of state.layers.filter(l=>l.type==='cells'&&(l===linked||dataView.source.combined&&l.combinedTable))){
        layer.columns=null;layer.live=null;layer.filters=[];layer.color_by=removed.has(layer.color_by)?'':renamed[layer.color_by]||layer.color_by;bump(layer);
      }
      dataView.info=await postJSON(dataRoute('info'),dataBody());dataView.selectedColumns=new Set();await dataLoad();
      await state.studio?.tablesChanged?.();
      buildPanel();message('Saved. The original is kept in previous/.');
    }else{
      dataView.busy=false;dataOpenFile({file_token:saved.file_token,name:saved.name});message('Saved '+saved.name+'.');
    }
    return true;
  }catch(error){dataView.lastError=error.message;dataView.status.textContent=error.message;message(error.message,true);return false;}
  finally{clearInterval(poll);dataView.busy=false;dataView.holder.classList.remove('data-busy');dataDrawHead();}
}

async function dataOpenUpload(file) {
  try {
    const found = await bringUpload(file);
    if (found.kind !== "table") throw Error(file.name + " is not a table.");
    dataRegisterFile(found);
    message(found.name + ' added. Use Change cell data to choose it; the current table is unchanged.');
  } catch (error) { message(error.message, true); }
}

// Registering never chooses a source, clears edits, or changes the image.
function dataRegisterFile(found) {
  dataView.opened = [...(dataView.opened || []).filter((o) => o.file !== found.file_token), { file: found.file_token, name: found.name, path:found.path }];
  if(dataView.head)dataDrawHead();
  buildPanel();
}

// Opening is a separate explicit user action (also used after Save as).
function dataOpenFile(found) {
  dataRegisterFile(found);
  showDataTab();
  dataDrawHead();
  dataChoose("file|" + found.file_token);
}

// The image's cells picked (click, Ctrl-click): their rows are selected, or filter the rows.
document.addEventListener("spacemap-cells-picked", async (event) => {
  const { layer, labels, fromData } = event.detail || {};
  if (!layer) return;
  dataView.groupPick = null;
  dataView.pickedLabels = labels || [];
  dataView.pickedRoi = dataLayerRoi(layer);
  dataView.pickedFromViewer = !fromData;
  dataView.pickedLayer = layer.token;
  if (!dataView.holder?.isConnected || !dataView.source || dataLinkedLayer() !== layer) return;
  if (dataView.onlySelected) { dataFiltersChanged(); return; }
  if (fromData) return;
  if (!labels?.length) { dataView.selected.clear(); dataDraw(); return; }
  if (!dataView.follow || dataLinkedLayer() !== layer) return;
  try {
    const found = await postJSON(dataRoute("find"), dataBody({ label: labels[labels.length - 1], cell_roi:dataLayerRoi(layer) }));
    if (found.offset === null) { dataView.status.append(el("span", { class: "muted", text: " · cell " + labels[labels.length - 1] + " is not in these rows" })); return; }
    dataView.offset = Math.floor(found.offset / 200) * 200;
    await dataLoad(true);
    const row = dataView.result.rows[found.offset - dataView.offset];
    if (row) { dataView.selected = new Set([row[0]]); dataDraw(); dataView.grid.querySelector("tr.selected")?.scrollIntoView({ block: "nearest" }); }
  } catch (error) { /* the table may not hold this cell */ }
});

// The Data tab's content, built once.
function dataViewInto(container) {
  if (dataView.holder?.isConnected) return;
  dataView.head = el("div", { class: "data-head" });
  dataView.grid = el("div", { class: "data-scroll", tabindex: "0" });
  dataView.status = el("div", { class: "data-status", role: "status" });
  dataView.holder = el("div", { class: "data-view" }, [dataView.head, dataView.grid, dataView.status]);
  dataView.grid.addEventListener("keydown", dataKeys);
  container.append(dataView.holder);
  dataDrawHead();
  if (!dataView.source) {
    const cells=state.layers.find(l=>l.type==='cells'&&l.id===state.cellDataLayer)||state.layers.find(l=>l.type==='cells');
    const key=cells?.table===cells?.combinedTable?'combined':cells?.id+'|'+cells?.table;
    const first = dataSources().find(item=>item.key===key)||dataSources()[0];
    if (first) dataChoose(first.key);
    else dataView.status.textContent = "No cell data available. Open cell data, or load an image with cells.";
  } else dataLoad(true);
  if(!dataView.combined)state.phenotypingLink=dataRefreshCombined().catch(error=>{dataView.status.textContent=error.message;});
}

// One checked combined input, available even when no segmentation/image is open.
async function dataRefreshCombined() {
  if(state.reviewer)return;
  if (!new URLSearchParams(location.search).get('studio')) return;
  clearTimeout(dataView.linkRetryTimer);
  consolidateCellResults();
  const generation = state.initGeneration;
  const choice = dataView.choiceSequence;
  const sequence = dataView.linkSequence = (dataView.linkSequence || 0) + 1;
  const layers = state.layers.filter(l => l.type === 'cells');
  const chosen = new Map(layers.map(layer => [layer, {table:layer.table, colour:layer.color_by}]));
  const found = await postJSON(dataRoute('query').replace('/data/query','/phenotyping_link'), {layers:layers.map(l => l.token)});
  if (generation !== state.initGeneration || sequence !== dataView.linkSequence) return;
  const previous = dataView.combined?.path;
  dataView.combined = found.table;
  if (previous !== found.table?.path) { dataView.groupPick = null; dataView.pickedLabels = []; dataView.selected.clear(); }
  const available=new Set(found.table?.columns||[]);
  if(choice===dataView.choiceSequence&&dataView.source?.combined&&found.table&&!dataDirtySources.has(dataView.sourceKey)){
    dataView.filters=dataView.filters.filter(item=>available.has(item.col));
    if(dataView.sort&&!available.has(dataView.sort.col))dataView.sort=null;
    if(dataView.columns)dataView.columns=dataView.columns.filter(name=>available.has(name));
    if(dataView.groupPick&&!available.has(dataView.groupPick.column)){dataView.groupPick=null;dataView.onlySelected=false;}
  }
  for (const layer of layers) {
    if (!state.layers.includes(layer)) continue;
    const previousSuggestions=JSON.stringify(suggestionSettings(layer));
    const link = found.layers[layer.token];
    // Identical saved data must not invalidate tiles, columns or density maps.
    // Aliases are deliberately excluded: binding adds the canonical alias on
    // the server, but that does not change any cell values on the next call.
    const linkStamp = JSON.stringify([found.table, link?.linked, link?.table, link?.keys, link?.reason, link?.retryable]);
    const unchanged = !!found.table?.revision && layer.combinedLinkStamp === linkStamp;
    layer.combinedTable = link?.linked ? link.table : null;
    layer.combinedLinkNote = link?.reason || '';
    layer.combinedLinkPending = !!link?.retryable;
    // The file watcher and job completion share the same revision. Whichever
    // sees it first refreshes the data; the other must not repaint it again.
    if (link?.linked && found.table?.revision)
      state.stamps[layer.token + '|' + link.table] = found.table.revision;
    if (unchanged) continue;
    layer.combinedLinkStamp = linkStamp;
    const sameChoice = layer.table === chosen.get(layer).table && layer.color_by === chosen.get(layer).colour;
    if (!link?.linked) {
      layer.tables = layer.tables.filter(t => !t.id.startsWith('merged_'));
      if (layer.table?.startsWith('merged_')) { layer.table = layer.tables[0]?.id || ''; layer.color_by = ''; layer.columns = null; layer.filters = []; bump(layer); }
      continue;
    }
    const fresh = !layer.tables.some(t => t.id === link.table);
    // The saved table can initially be listed under a local layer alias while
    // its phenotyping link prepares. Follow that same file's canonical link
    // without resetting the user's draft, filters, columns or image scope.
    if (choice===dataView.choiceSequence && !dataView.busy && dataView.source?.cells===layer.token &&
        (link.aliases||[]).includes(dataView.source.table)) {
      const dirty=dataDirtySources.delete(dataView.sourceKey);
      dataView.source={combined:true};dataView.sourceKey='combined';
      if(dirty)dataDirtySources.add('combined');
      if(dataView.rows.this_image)dataView.rows={...dataView.rows,column:found.table.roles.roi,value:link.roi};
      dataView.loadSequence=(dataView.loadSequence||0)+1;
    }
    layer.tables = layer.tables.filter(t => t.id === link.table || (!(link.aliases || []).includes(t.id) && !t.id.startsWith('merged_')));
    if (fresh) layer.tables.push({id:link.table,name:'Phenotyping results',file:found.table?.name});
    layer.table_roles = link.table_roles;
    layer.density_tables=[...layer.tables,...(layer.density_tables||[]).filter(t=>t.id.startsWith("p"))];
    if (sameChoice && (fresh || previous !== found.table?.path || !layer.tables.some(t => t.id === layer.table))) {
      layer.table = link.table; layer.live = null;
    }
    invalidateCellColumns(layer);
    if(layer.table===link.table){
      if(layer.color_by&&!layer.color_by.startsWith('preview:')&&!available.has(layer.color_by)){
        layer.color_by='';layer.hidden=new Set();layer.range=null;
      }
      layer.filters=(layer.filters||[]).filter(item=>available.has(item.col));
      layer.label_columns=(layer.label_columns||[]).filter(name=>available.has(name));
    }
    if(layer.table===link.table)restoreReferenceChoice(layer,found.table.path);
    if(state.inspectedCell?.layer===layer&&layer.suggestionsOn&&
        previousSuggestions!==JSON.stringify(suggestionSettings(layer)))
      requestCellProfile(layer,state.inspectedCell.result,++state.cardSeq);
    if (layer.table === link.table) bump(layer, {keepVisible:true});
  }
  if (Object.values(found.layers).some(link => link.retryable)) {
    dataView.linkRetryTimer = setTimeout(() => {
      if (generation !== state.initGeneration || sequence !== dataView.linkSequence) return;
      state.phenotypingLink = dataRefreshCombined().catch(error => message(error.message, true));
    }, 2500);
  }
  buildPanel();
  document.querySelectorAll('.manual-reference-card').forEach(card=>card.refreshSelection?.());
  if (dataView.holder?.isConnected && choice===dataView.choiceSequence) {
    try {
    if (!dataView.source || (dataView.source.combined && previous !== found.table?.path)) await dataChoose('combined');
    else if (dataView.source.combined) {
      const linked = dataLinkedLayer();
      if(linked&&dataView.awaitingImageScope){
        dataView.rows={this_image:true,column:found.table.roles.roi,value:dataLayerRoi(linked)};
        dataView.awaitingImageScope=false;
      }
      if (linked && dataView.rows.this_image) dataView.rows.value = dataLayerRoi(linked);
      const info = await postJSON(dataRoute('info'), dataBody());
      if (generation!==state.initGeneration || sequence!==dataView.linkSequence || choice!==dataView.choiceSequence) return;
      if (dataPreparing(info,()=>dataRefreshCombined())) return;
      dataView.info = info;
      dataDrawHead(); await dataLoad(true);
    }
    else { dataDrawHead(); await dataLoad(true); }
    } catch(error) { if(choice===dataView.choiceSequence)dataReadError(error); }
  }
}

document.addEventListener('spacemap-groups-picked', async event => {
  const {layer,column,values} = event.detail;
  const generation=state.initGeneration, selectionSequence=state.groupPickSequence;
  const pickVersion=dataView.groupPickVersion=(dataView.groupPickVersion||0)+1;
  const filtered = event.detail.filtered ?? values.length > 0;
  dataView.groupPick = filtered ? {column,values} : null;
  dataView.onlySelected = filtered;
  dataView.pickedLabels = []; dataView.selected.clear();
  if (!dataView.holder?.isConnected) return;
  if (layer && layer.table!==layer.combinedTable) {
    const key=layer.id+'|'+layer.table;
    if(dataView.sourceKey!==key)await dataChoose(key);
  } else if (dataView.combined && !dataView.source?.combined) await dataChoose('combined');
  if(pickVersion!==dataView.groupPickVersion||generation!==state.initGeneration||selectionSequence!==state.groupPickSequence)return;
  // An explicit source choice clears old selections. Restore this new group's
  // selection only after its required source is ready, and only if still current.
  const expected=layer?(layer.table===layer.combinedTable?'combined':layer.id+'|'+layer.table):null;
  if(expected&&dataView.sourceKey!==expected)return;
  dataView.groupPick=filtered?{column,values}:null;dataView.onlySelected=filtered;
  if (dataView.source?.combined && layer) dataView.rows = {...dataView.rows, value:dataLayerRoi(layer)};
  await dataLoad();
  if (pickVersion===dataView.groupPickVersion && dataView.onImage) dataShowOnImage(true);
});

// Compact editing tools; every change is staged until Save or Save as.
function dataEditorMenu(event, column, row) {
  if(state.reviewer){event.preventDefault();return;}
  event.preventDefault();event.stopPropagation();if(dataView.busy)return;
  closeContextMenu();
  if(row!==undefined){dataView.pickedFromViewer=false;if(!dataView.selected.has(row))dataView.selected=new Set([row]);}
  else if(!dataView.selectedColumns?.has(column))dataView.selectedColumns=new Set([column]);
  dataDraw();
  const menu=el('div',{id:'ctxmenu',class:'data-actions-menu',role:'menu','aria-label':row===undefined?'Column actions':'Row actions'});
  const add=(text,action)=>menu.append(el('button',{class:'btn small',role:'menuitem',text,onclick:()=>{closeContextMenu();action();}}));
  if(row===undefined){
    menu.append(el('strong',{text:dataView.selectedColumns.size>1?dataView.selectedColumns.size+' columns':column}));
    if(dataView.selectedColumns.size===1){
      add('Rename column…',()=>dataEditDialog({kind:'rename_column',column}));
      add('Replace value…',()=>dataEditDialog({kind:'replace',column}));
      add('Fill column…',()=>dataEditDialog({kind:'fill',column}));
      add('Sort ascending',()=>{dataView.sort={col:column,desc:false};dataLoad();});
      add('Sort descending',()=>{dataView.sort={col:column,desc:true};dataLoad();});
    }
    add('Hide columns',()=>{dataView.columns=dataView.result.shown.filter(c=>!dataView.selectedColumns.has(c));dataLoad(true);});
    add('Delete columns (whole dataset)',()=>dataMutate({kind:'delete_columns',columns:[...dataView.selectedColumns]}));
  }else{
    menu.append(el('strong',{text:dataView.selected.size+' selected rows'}));
    if(column){add('Fill selected values…',()=>dataEditDialog({kind:'fill',column,scope:'selected'}));add('Clear selected values',()=>dataMutate({kind:'fill',column,scope:'selected',value:null}));}
    add('Delete selected rows',()=>dataMutate({kind:'delete_rows',scope:'selected'}));
    add('Delete filtered rows…',()=>dataEditDialog({kind:'delete_rows',scope:'filtered'}));
  }
  dataDocument().body.append(menu);const rect=menu.getBoundingClientRect();
  menu.style.left=Math.max(8,Math.min(event.clientX,dataDocument().defaultView.innerWidth-rect.width-8))+'px';menu.style.top=Math.max(8,Math.min(event.clientY,dataDocument().defaultView.innerHeight-rect.height-8))+'px';
  menu.querySelector('button')?.focus();
}

function dataEditDialog(operation) {
  closeContextMenu();
  const labels={rename_column:'Rename column',replace:'Replace value',fill:'Fill values',delete_rows:'Delete rows',save_as:'Save as'};
  const dialog=el('dialog',{class:'data-edit-dialog','aria-label':labels[operation.kind]});
  const form=el('form',{method:'dialog'}),title=el('strong',{text:labels[operation.kind]+(operation.column?' · '+operation.column:'')});
  const status=el('div',{class:'hint',role:'status'});form.append(title);
  let old=null,value=null,scope=null;
  if(operation.kind==='replace'){
    old=el('input',{type:'text',value:operation.old??'','aria-label':'Find exact value'});form.append(field('Find',old));
  }
  if(operation.kind!=='delete_rows'){
    const initial=operation.kind==='rename_column'?operation.column:operation.kind==='save_as'?(dataView.info?.name||'table.parquet').replace(/(\.[^.]+)$/,'_edited$1'):'';
    value=el('input',{type:'text',value:initial,'aria-label':operation.kind==='save_as'?'Save file name':operation.kind==='rename_column'?'New column name':'New value'});
    form.append(field(operation.kind==='save_as'?'File name':operation.kind==='rename_column'?'Name':'Replace with',value));
  }
  if(['replace','fill','delete_rows'].includes(operation.kind)){
    scope=select([['filtered','Filtered rows · '+(dataView.result?.matched||0).toLocaleString()],['all','Entire dataset'],['selected','Selected rows · '+dataView.selected.size]],operation.scope||'filtered',()=>{});
    scope.setAttribute('aria-label','Edit scope');scope.querySelector('[value="selected"]').disabled=!dataView.selected.size;
    form.append(field('Apply to',scope));
    if(operation.kind==='delete_rows')status.textContent='Rows are removed from the table. Undo is available until Save.';
    else status.textContent='Exact values only. Changes stay in Data until Save.';
  }
  const cancel=el('button',{class:'btn small',type:'button',text:'Cancel',onclick:()=>dialog.close()});
  const apply=el('button',{class:'btn small primary',type:'submit',text:operation.kind==='save_as'?'Save':operation.kind==='delete_rows'?'Delete rows':'Apply'});
  form.append(status,el('div',{class:'row data-dialog-buttons'},[cancel,apply]));dialog.append(form);dataDocument().body.append(dialog);
  dialog.addEventListener('close',()=>dialog.remove());
  form.onsubmit=async event=>{
    event.preventDefault();apply.disabled=true;
    const op={...operation};if(scope)op.scope=scope.value;
    if(operation.kind==='rename_column')op.name=value.value;
    else if(operation.kind==='save_as')op.name=value.value;
    else if(value)op.value=value.value;
    if(old)op.old=old.value;
    const ok=operation.kind==='save_as'?await dataSave('copy',op.name):await dataMutate(op);
    if(ok)dialog.close();else {status.textContent=dataView.lastError||'Could not apply this change.';apply.disabled=false;}
  };
  dialog.showModal();(value||scope)?.focus();value?.select();
}

function dataEditorSnapshot(){return JSON.parse(JSON.stringify({filters:dataView.filters,sort:dataView.sort,columns:dataView.columns,rows:dataView.rows}));}
function dataEditorSchema(found){
  const previous=dataView.info?.renamed||{},inverse=Object.fromEntries(Object.entries(previous).map(([a,b])=>[b,a]));
  const remap=name=>{const original=inverse[name]||name;return found.removed?.includes(original)?null:found.renamed?.[original]||original;};
  dataView.filters=dataView.filters.map(f=>({...f,col:remap(f.col)})).filter(f=>found.columns.includes(f.col));
  if(dataView.sort){dataView.sort={...dataView.sort,col:remap(dataView.sort.col)};if(!found.columns.includes(dataView.sort.col))dataView.sort=null;}
  if(dataView.columns)dataView.columns=dataView.columns.map(remap).filter(c=>found.columns.includes(c));
  if(dataView.rows.column)dataView.rows.column=remap(dataView.rows.column);
  dataView.selectedColumns=new Set([...(dataView.selectedColumns||[])].map(remap).filter(c=>found.columns.includes(c)));
}
async function dataMutate(operation, selectionFilter = null) {
  if(state.reviewer)return;
  if(dataView.busy||!dataView.result)return false;
  const before=dataEditorSnapshot(),sourceKey=dataView.sourceKey;
  dataView.busy=true;dataView.lastError='';dataView.holder.classList.add('data-busy');dataDrawHead();dataView.status.textContent='Applying change…';
  try{
    const found=await postJSON(dataRoute('mutate'),dataBody({operation:selectionFilter ? {...operation,scope:'filtered'} : operation,
      selected:[...dataView.selected],revision:dataView.result.revision,...(selectionFilter || {})}));
    if(dataView.sourceKey!==sourceKey)return false;
    dataView.lastAffected=found.affected;
    dataEditorSchema(found);
    if(operation.kind==='replace'){
      const filter=dataView.filters.find(f=>f.col===operation.column);
      if(filter?.values?.includes(operation.old))filter.values=[...new Set(filter.values.flatMap(v=>v===operation.old?(operation.scope==='selected'?[v,operation.value]:[operation.value]):[v]))];
    }
    if(found.affected){(dataView.undoViews||=[]).push(before);dataView.redoViews=[];}
    if(operation.kind==='delete_rows')dataView.selected.clear();
    dataView.info={...dataView.info,...found};dataView.result={...dataView.result,...found};
    dataView.lastEditNotice=operation.kind==='delete_columns'?found.affected.toLocaleString()+' columns deleted across the dataset. Save to keep.':
      found.affected.toLocaleString()+(operation.kind.includes('column')?' columns changed':' values / rows changed');
    await dataLoad(true);return true;
  }catch(error){dataView.lastError=error.message;dataReadError(error);return false;}
  finally{dataView.busy=false;dataView.holder.classList.remove('data-busy');dataDrawHead();}
}
async function dataHistory(action) {
  if(state.reviewer)return;
  if(dataView.busy)return;
  if(action==='discard'&&!dataDocument().defaultView.confirm('Discard all unsaved Data edits for this table?'))return;
  dataView.busy=true;dataView.holder.classList.add('data-busy');dataDrawHead();
  try{
    const found=await postJSON(dataRoute(action),dataBody());
    const from=action==='undo'?dataView.undoViews:action==='redo'?dataView.redoViews:null;
    const to=action==='undo'?(dataView.redoViews||=[]):(dataView.undoViews||=[]);
    const snapshot=from?.pop();if(snapshot){to.push(dataEditorSnapshot());Object.assign(dataView,snapshot);}else if(action==='discard'){dataView.filters=[];dataView.sort=null;dataView.columns=null;dataView.undoViews=[];dataView.redoViews=[];}
    dataEditorSchema(found);dataView.info={...dataView.info,...found};dataView.result={...dataView.result,...found};
    dataView.selected.clear();dataView.lastEditNotice='';await dataLoad(true);
  }catch(error){message(error.message,true);}
  finally{dataView.busy=false;dataView.holder.classList.remove('data-busy');dataDrawHead();}
}
