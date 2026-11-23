/* Shared language and connections for cell overlays, Data and density maps. */
'use strict';

// A finished tagging job adds measurements to the current cells; it never opens
// another image or reinitializes image tiles. A later user choice wins.
async function refreshTaggedCells(outputs) {
  const layer = state.layers.find(l => l.type === 'cells' && l.id === state.cellDataLayer) || state.layers.find(l => l.type === 'cells');
  if (!layer) return false;
  const generation = state.initGeneration, table = layer.table, colour = layer.color_by,
    choice = state.groupPickSequence, request = layer.taggedRequest = (layer.taggedRequest || 0) + 1;
  const current = () => generation === state.initGeneration && state.layers.includes(layer) &&
    request === layer.taggedRequest && choice === state.groupPickSequence && layer.table === table && layer.color_by === colour;
  const session = new URLSearchParams(location.search).get('session');
  let result;
  for (let attempt = 0; attempt < 60 && current(); attempt++) {
    result = await postJSON('/studio/' + encodeURIComponent(session) + '/tagging_result',
      {token:layer.token, outputs, version:layer.taggedOutputVersion});
    if (!current()) return false;
    if (!result.retryable) break;
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  if (!current() || !result?.linked) return false;
  // Data edits must stay on their original source until the user saves them.
  if (dataLinkedLayer() === layer && dataDirtySources.has(dataView.sourceKey)) return false;
  layer.tables = (layer.tables || []).filter(t => t.id !== result.table && !(result.aliases || []).includes(t.id));
  layer.tables.push({id:result.table, name:result.name, file:result.file});
  layer.table_roles = result.table_roles;
  layer.table = result.table; layer.taggedOutputVersion = result.version;
  const savedColour = String(colour || '').replace(/^preview:/, '');
  if (String(colour || '').startsWith('preview:') && result.columns.includes(savedColour)) layer.color_by = savedColour;
  state.cellDataLayer = layer.id;
  invalidateCellColumns(layer);
  bump(layer, {keepVisible:true});
  await dataRefreshTaggedTable(layer, table);
  return true;
}

function cellDataInfo(layer, table = layer.table) {
  const item = (layer.tables || []).find(t => t.id === table) || {};
  const name = item.name || '';
  if (table && (table === layer.combinedTable || table.startsWith('merged_') || /combined phenotyping/i.test(name)))
    return {name:'Phenotyping results', detail:'Measurements, phenotype labels and clusters from Phenotyping. Only this image’s cells are drawn.', file:item.file || dataView.combined?.name || ''};
  if (/Tagged cells|Cell[- ]tagging results/i.test(name))
    return {name:'Cell tagging results', detail:'Measurements with the labels added in Cell tagging, for this segmentation.', file:item.file || ''};
  if (/This segmentation.s cells|Segmentation measurements/i.test(name))
    return {name:'Segmentation measurements', detail:'Original per-cell measurements from Segmentation. Cell outlines come from the segmentation file.', file:item.file || ''};
  return {name:name.replace(/^Data: /,'Imported cell data: ') || 'No cell data available', detail:table?'Measurements linked to these cells.':'Add cell data, or run Segmentation to create per-cell measurements.', file:item.file || ''};
}

function cellControlRow(label, parts, tip) {
  return el('div',{class:'row field cell-line'},[el('label',{class:'lbl',text:label}),
    el('div',{class:'inline grow wrap'},parts.filter(Boolean)),tip?help(tip):null].filter(Boolean));
}

function densityCellLayer(layer) {
  return state.layers.find(c=>c.type==='cells'&&c.id===layer.cellsId&&c.image_token===layer.image_token)||
    state.layers.find(c=>c.type==='cells'&&c.token===layer.token);
}

function resolveDensityData(layer) {
  const cells=densityCellLayer(layer);
  if (!cells) return null;
  layer.cellsId=cells.id;layer.token=cells.token;layer.image_token=cells.image_token;
  // Cell-data links can be added after maps have been restored from a saved view.
  const extra=(cells.density_tables||[]).filter(t=>t.id.startsWith('p'));
  layer.tables=[...(cells.tables||[]),...extra];layer.combinedTable=cells.combinedTable;
  // Legacy point/triad maps have no segmentation counterpart. Cell maps always
  // use the one shared Cell data choice, including maps saved by older versions.
  layer.dataMode=layer.dataMode==='snapshot'?'snapshot':layer.table?.startsWith('p')?'fixed':'viewer';
  if(layer.dataMode==='fixed')layer.cellScope='all';
  if (layer.dataMode==='viewer') layer.table=cells.table || '';
  layer.dataMissing=!layer.tables.some(t=>t.id===layer.table);
  if (layer.dataMissing && layer.dataMode!=='snapshot') layer.table='';
  return cells;
}

// One active geometry/table pair, independent of visibility and layer order.
function activeCellLayer(layers = state.layers) {
  const cells=layers.filter(l=>l.type==='cells'&&!l.selection&&!isTrainingCellLayer(l));
  return cells.find(l=>l.id===state.cellDataLayer)||cells.find(l=>l===dataLinkedLayer())||cells.find(l=>l.visible)||cells[0]||null;
}

async function chooseCellSource(layer, table = layer.table, reset = true) {
  if (!state.layers.includes(layer)) return;
  if (dataDirtySources.size) { message('Save or discard your Data edits before changing Cell data or outlines.', true); buildPanel(); return; }
  if (state.inspectedCell) clearCellInspection({notify:false});
  dataView.filters=[];dataView.groupPick=null;dataView.onlySelected=false;dataView.pickedLabels=[];dataView.selected.clear();
  state.cellDataLayer=layer.id;
  if(!reset&&!layer.visible)setVisible(layer,true);
  claimCellView(layer,'Cell data changed');
  if (reset) {layer.table=table;layer.color_by='';layer.hidden=new Set();layer.filters=[];layer.range=null;layer.live=null;invalidateCellColumns(layer);}
  bump(layer);
  if (dataView.holder?.isConnected && layer.table) await dataChoose(layer.table===layer.combinedTable?'combined':layer.id+'|'+layer.table);
}

function sharedCellDataControl(layers) {
  const cells=layers.filter(l=>l.type==='cells'&&!l.selection&&!isTrainingCellLayer(l)),box=el('div',{class:'shared-cell-data','data-shared-cell-data':''});
  if(!cells.length){box.append(el('span',{class:'hint',text:'No cells loaded'}),help('Open an image with segmentation, or add cell outlines below.'));return box;}
  const current=activeCellLayer(cells);
  const available=!!current.table && (current.tables||[]).some(t=>t.id===current.table);
  const options=(current.tables||[]).map(t=>[current.id+'|'+t.id,cellDataInfo(current,t.id).name]);
  if(!available)options.unshift(['',options.length?'Choose cell data…':'No cell data available']);
  const pick=select(options,available?current.id+'|'+current.table:'',async value=>{
    const [id,table]=value.split('|'),layer=cells.find(l=>l.id===id);if(!layer)return;
    await chooseCellSource(layer,table);
  });pick.setAttribute('aria-label','Cell data');pick.disabled=!options.some(([value])=>value);
  const info=available?cellDataInfo(current):null;
  const tip=[info?.detail,info?.file?'File: '+info.file:'',
    'Shared by cell overlays, density maps and Data. Cell outlines come from Segmentation.',
    'Flow gating and cluster clicks use Phenotyping results.',
    ...(current.hidden_tables||[]).map(h=>h.name+': '+h.reason)].filter(Boolean).join('\n\n');
  box.append(cellControlRow('Cell data',[pick,
    el('button',{class:'btn small',text:'Open data',disabled:available?null:'',onclick:()=>openCellData(current)})],tip));
  return box;
}

function syncDensityMaps(cells) {
  for (const map of state.layers.filter(m=>m.type==='density' && (m.token===cells.token || m.cellsId===cells.id))) {
    const before=map.table;resolveDensityData(map);
    if (before!==map.table) {map.fcol='';map.fvals=[];map.vcol='';invalidateCellColumns(map);}
    if (before!==map.table || map.cellScope==='viewer') {
      state.versions[map.id]=(state.versions[map.id]||0)+1;refreshLayer(map);
    }
  }
}

// A direct interaction replaces the previous selection owner. A stale Flow
// mask must never override a later cluster click or Data filter.
function claimCellView(layer, source) {
  state.groupPickSequence=(state.groupPickSequence||0)+1;dataView.groupPickVersion=(dataView.groupPickVersion||0)+1;
  if (layer.flowSelection || dataView.flowSelection) window.clearFlowGateView?.({silent:true});
  layer.selection=null;layer.viewSource=source;
}

function resetCellView(layer) {
  claimCellView(layer,'All cells');layer.filters=[];layer.hidden=new Set();layer.range=null;
  if (dataLinkedLayer()===layer) {
    dataView.filters=[];dataView.onlySelected=false;dataView.groupPick=null;dataView.selected.clear();
    if(dataView.holder?.isConnected){dataDrawHead();dataLoad();}
  }
  bump(layer);
}

function cellViewStatus(layer) {
  const filtered=layer.flowSelection || layer.selection || layer.filters?.length || layer.hidden?.size || layer.range;
  let text='All cells in this image';
  if(layer.flowSelection)text=(layer.viewSource||'Flow gating')+' · '+(layer.flowPopulation?.name||'Selected population')+' · '+Number(layer.flowPopulation?.by_image?.[dataLayerRoi(layer)]||0).toLocaleString()+' cells in this image';
  else if(filtered)text=(layer.viewSource||'Viewer filters')+(layer.color_by?' · '+cellsPreviewName(layer.color_by):'');
  const clear=el('button',{class:'btn small',text:'Show all cells','aria-label':'Show all image cells',onclick:()=>resetCellView(layer)});clear.disabled=!filtered;
  return el('div',{class:'cell-view-status','data-cell-view-status':''},[
    cellControlRow('Cells shown',[el('span',{class:'grow',text}),clear],
      'Filters affect the cell overlay. Image markers remain visible.')]);
}

function openCellData(layer) {
  showDataTab();return dataChoose(layer.table===layer.combinedTable?'combined':layer.id+'|'+layer.table);
}

async function showViewerCellGroups(column, shown, all, context = {}) {
        const sequence = state.groupPickSequence = (state.groupPickSequence || 0) + 1;
        const generation = state.initGeneration;
        // A command still running (a group's top markers, say) returns the view it was sent with: let it land
        // first, so the groups asked for now are not put back to the earlier ones.
        await (state.commandQueue || Promise.resolve()).catch(() => {});
        await state.phenotypingLink;
        if (sequence !== state.groupPickSequence || generation !== state.initGeneration) return false;
        consolidateCellResults();
        if (context.table && context.table !== dataView.combined?.path) return false;
        const cells = state.layers.find(item=>item.type==='cells'&&item.combinedTable&&item.id===state.cellDataLayer) || state.layers.find(item => item.type === 'cells' && item.combinedTable) || (!dataView.combined && state.layers.find(item => item.type === 'cells' && (item.tables || []).length));
        const table=cells?await cellTableWith(cells,[column],cells.combinedTable,false):null;
        if (sequence !== state.groupPickSequence || generation !== state.initGeneration) return false;
        if (!cells || !table) {
          document.dispatchEvent(new CustomEvent('spacemap-groups-picked', {detail:{layer:null,column,values:shown || []}}));
          return false;
        }
        // Include unassigned cells and groups outside this run in the hidden
        // set, so picking one subset really isolates it.
        let groups = all;
        if (shown?.length) {
          const legend = await getJSON('/cells/' + cells.token + '/legend?t=' + encodeURIComponent(table) + '&col=' + encodeURIComponent(column));
          groups = [...new Set([...all, ...(legend.items || []).map(item => item.value)])];
        }
        if (sequence !== state.groupPickSequence || generation !== state.initGeneration || !state.layers.includes(cells)) return false;
        state.cellDataLayer=cells.id;claimCellView(cells,"Phenotyping clusters");
        if(cells.table!==table){cells.table=table;invalidateCellColumns(cells);}
        cells.live = null; cells.color_by = column;
        cells.filters = shown?.length ? [{col:column,values:shown.map(String)}] : [];
        cells.range = null;
        cells.hidden = new Set(shown && shown.length ? groups.filter(value => !shown.includes(value)) : []);
        if ((cells.mode || 'outline') === 'outline') cells.mode = 'both';
        if (!cells.visible) setVisible(cells, true);
        bump(cells);
        document.dispatchEvent(new CustomEvent('spacemap-groups-picked', {detail:{layer:cells,column,values:shown || []}}));
        return true;
}

// Only maps made by the Cells button belong to its toggle. Other workflows can
// still open independent maps. Recognize the previous button's saved maps too.
function isCellDensityMap(layer) {
  return layer.type==='density' && (layer.cellDensity || (layer.cellsId && layer.dataMode==='viewer' &&
    layer.cellScope==='viewer' && (layer.name==='Density: shown cells' || (layer.vcol && layer.name==='Mean '+layer.vcol))));
}

// Retain saved map settings, but opening an image never turns its Cells density
// on. The explicit Density button restores it; folding Cells doesn't change it.
function restoreCellDensityMaps() {
  if(!state.studio)return;
  state.layers.filter(isCellDensityMap).forEach(map=>{
    map.cellDensity=true;map.densityDormant=true;map.visible=false;
  });
}

// Toggle maps only from the active viewer source; asynchronous metadata cannot
// replace a later source/group choice. Count and local mean remain distinct.
function densityFromCellsButton(cells) {
  const maps=()=>state.layers.filter(m=>isCellDensityMap(m)&&densityCellLayer(m)===cells);
  const enabled=maps().some(m=>!m.densityDormant);
  const button=el('button',{class:'btn small cell-density-toggle',text:'Density','aria-label':'Density from shown cells','aria-pressed':String(enabled),
    title:'Turn the density map on or off. Groups use cell density; numeric columns use their local mean. Change Map value to choose explicitly.'});
  button.disabled=!!cells.densityPending || (!enabled&&(!cells.table || !!cells.pending));
  button.onclick=async()=>{
    if(cells.densityPending)return;
    const existing=maps();
    if(existing.some(m=>!m.densityDormant)){existing.forEach(map=>removeLayer(map));return;}
    if(cells.live || String(cells.color_by||'').startsWith('preview:')){message('Save the preview results before making a density map.',true);return;}
    if(existing.length){
      existing.forEach(map=>{map.densityDormant=false;cellRowsOpen()[map.id]=true;resolveDensityData(map);setVisible(map,true);});
      buildPanel();return;
    }
    const generation=state.initGeneration,table=cells.table,column=cells.color_by,sequence=state.groupPickSequence;
    const current=()=>generation===state.initGeneration&&state.layers.includes(cells)&&cells===activeCellLayer()&&table===cells.table&&column===cells.color_by&&sequence===state.groupPickSequence;
    cells.densityPending=true;button.disabled=true;
    try{
      const legend=column?await getJSON('/cells/'+cells.token+'/legend?'+cellColourQuery(cells)):null;
      if(!current())return;
      const numeric=legend?.kind==='numeric';
      const groups=(legend?.items||[]).filter(item=>!cells.hidden?.has(item.value));
      const map={id:'density-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2,7),type:'density',cellDensity:true,
        name:numeric?'Mean '+column:'Density: shown cells',
        token:cells.token,image_token:cells.image_token,cellsId:cells.id,dataMode:'viewer',cellScope:'viewer',
        tables:cells.tables,table:cells.table,fcol:'',fvals:[],vcol:numeric?column:'',radius:50,weighted:false,
        cmap:numeric?'viridis':groups.length===1?groups[0].color:'#ff3b3b',cmapAuto:true,opacity:.75};
      cellRowsOpen()[map.id]=true;
      await addLayerToView(map,cells);
    }catch(error){message(error.message,true);}finally{
      cells.densityPending=false;
      if(generation===state.initGeneration&&state.layers.includes(cells))buildPanel();
    }
  };
  return button;
}

function setDensityScope(layer,value) {
  const owner=densityCellLayer(layer);
  if(value==='snapshot'){
    if(!owner || owner.live || String(owner.color_by||'').startsWith('preview:')){message('Save the preview results before keeping this selection.',true);buildPanel();return;}
    layer.densitySnapshot={filters:JSON.parse(JSON.stringify(owner.filters||[])),color_by:owner.color_by||'',
      hidden:Array.from(owner.hidden||[]),range:owner.range?.slice()||null,flowSelection:owner.flowSelection||'',
      flowMode:owner.flowMode||'filter',selection:owner.selection||''};
    layer.dataMode='snapshot';layer.table=owner.table;layer.cellScope='snapshot';
  }else{
    layer.cellScope=value;delete layer.densitySnapshot;
    if(owner){layer.dataMode='viewer';resolveDensityData(layer);}
  }
  if(!layer.vcol&&!layer.fvals?.length)layer.name=value==='snapshot'?'Density: kept selection':value==='all'?'Density: all cells':'Density: shown cells';
  bump(layer);
}

// Imports are registered without replacing the displayed geometry or table.
async function registerImportedCells(spec) {
  if(!state.layers.some(l=>l.type==='image'&&l.token===spec.image_token))return null;
  const active=activeCellLayer(),existing=state.layers.find(l=>l.id===spec.id);
  if(active)state.cellDataLayer=active.id;
  if(existing){
    if(existing.token!==spec.token){
      // Registry tokens may change when a source is rebuilt. Never transplant
      // its table IDs onto another token, where they could name different files.
      const sameSource=state.layers.find(l=>l.type==='cells'&&l.token===spec.token);
      if(sameSource)return sameSource;
      return addLayerToView({...spec,id:spec.id+'-'+spec.token.slice(-8),visible:false});
    }
    const tables=new Map((existing.tables||[]).map(t=>[t.id,t]));
    (spec.tables||[]).forEach(t=>tables.set(t.id,t));existing.tables=[...tables.values()];
    existing.table_roles={...(existing.table_roles||{}),...(spec.table_roles||{})};
    invalidateCellColumns(existing);buildPanel();saveView();if(dataView.head)dataDrawHead();return existing;
  }
  return addLayerToView({...spec,visible:!active});
}
