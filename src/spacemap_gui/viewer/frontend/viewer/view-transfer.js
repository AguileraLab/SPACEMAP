/* Carry display choices between images in one window, never data or geometry. */
'use strict';

function displayScope(spec) {
  return state.reviewer ? state.apiBase : spec?.display_scope || spec?.session || null;
}
function displayFamily(layer) {
  if(layer.review_inline)return 'review';
  if(layer.type==='image')return layer.id==='he'?'he':'image:'+layer.id;
  if(state.reviewer&&layer.type==='cells'&&layer.id.startsWith('review-cells-'))return 'shared-cells';
  if(layer.type==='annotations'&&layer.annotation_display_family)return 'annotations:'+layer.annotation_display_family;
  if(layer.type==='shapes'&&(layer.working_tissue||layer.annotation_kind==='tissue'||layer.artifact_part||layer.training_part))
    return JSON.stringify(['working',layer.id,layer.artifact_part||'',layer.training_part||'']);
  const path=layer.saved_file||layer.source||'';
  const folder=path.includes('/')?path.slice(0,path.lastIndexOf('/')):'';
  // A different segmentation/import folder is a different source. Never bind
  // a similarly named old table to the newly opened geometry.
  return JSON.stringify([layer.type,layer.annotation_group||'',layer.annotation_kind||'',
    layer.artifact_part||'',layer.training_part||'',folder||layer.id]);
}
function displayTable(layer) {
  const table=layer.tables?.find(t=>t.id===layer.table);
  if(!table)return null;
  const file=table.source_file||table.file||'';
  return {id:table.id,name:cellDataInfo(layer).name,file,
    combined:layer.table===layer.combinedTable,combinedPath:layer.table===layer.combinedTable?dataView.combined?.path:null};
}
function annotationDisplayFingerprint(layer) {
  return JSON.stringify([layer.visible,[...(layer.hidden||[])].sort(),[...(layer.hiddenNames||[])].sort()]);
}
function displayAnnotationNames(layer) {
  // Retain the requested names through an image where none of them exist.
  // An explicit subsequent toggle takes precedence over this carried choice.
  const choice=layer.displayNameChoice;
  if(choice && choice.fingerprint===annotationDisplayFingerprint(layer))return choice.names;
  if(layer.type==='annotations'&&layer.annotationLegend)return layer.annotationLegend.items.filter(i=>!layer.hidden.has(i.value)).map(i=>i.value);
  if(layer.type==='shapes')return [...namesOf(layer)].filter(n=>!layer.hiddenNames?.has(n));
  return null;
}
function finishAnnotationDisplayNames(layer, legend) {
  const pending=layer.displayNamesPending;
  if(!pending)return;
  delete layer.displayNamesPending;
  if(pending.fingerprint===annotationDisplayFingerprint(layer)){
    layer.hidden=new Set(legend.items.filter(i=>!pending.names.includes(i.value)).map(i=>i.value));
    layer.displayNameChoice={names:pending.names,fingerprint:annotationDisplayFingerprint(layer)};
  }
  if(state.viewReady&&state.layers.includes(layer)){refreshLayer(layer);saveView();syncAnnotationVisibility();}
}
function displayLayout(layer, previous) {
  const blocks=state.annotationBlocksOpen||{}, source=layer.type==='annotations'||annotationSource(layer)?annotationFileKey(layer):'';
  const prefixed=(values,prefix)=>Object.fromEntries(Object.entries(values||{}).filter(([key])=>key.startsWith(prefix)).map(([key,value])=>[key.slice(prefix.length),value]));
  return {open:layer.open, autoOptionsOpen:layer.autoOptionsOpen,
    sourceOpen:blocks[source]??blocks[layer.name]??(layer.type==='shapes'&&!source),
    cellsOpen:blocks['cells::'+layer.id], rowOpen:state.cellRowsOpen?.[layer.id],
    rows:prefixed(state.annotationRowsOpen,source+'::'),roles:prefixed(state.annotationRolesOpen,source+'::'+layer.id+'::'),
    components:{...previous?.components,...Object.fromEntries(Object.entries(state.componentLists?.[layer.token]?.classes||{}).map(([name,memo])=>[name,{open:memo.open,sort:memo.sort}]))}};
}
function applyDisplayLayout(layer, layout) {
  if(!layout)return;
  for(const key of ['open','autoOptionsOpen'])if(layout[key]!==undefined)layer[key]=layout[key];
  const blocks=state.annotationBlocksOpen||(state.annotationBlocksOpen={});
  if(layout.cellsOpen!==undefined)blocks['cells::'+layer.id]=layout.cellsOpen;
  if(layout.rowOpen!==undefined)cellRowsOpen()[layer.id]=layout.rowOpen;
  if(!isAnnotationDisplayLayer(layer))return;
  const source=layer.type==='annotations'||annotationSource(layer)?annotationFileKey(layer):'';
  if(layout.sourceOpen!==undefined)blocks[source]=layout.sourceOpen;
  for(const [name,value] of Object.entries(layout.rows||{}))(state.annotationRowsOpen||={})[source+'::'+name]=value;
  for(const [name,value] of Object.entries(layout.roles||{}))(state.annotationRolesOpen||={})[source+'::'+layer.id+'::'+name]=value;
  if(layer.type==='annotations'){
    // No selected IDs, pagination or polygons cross to the new image.
    const memory=(state.componentLists||={})[layer.token]||={classes:{}};
    for(const [name,memo] of Object.entries(layout.components||{}))memory.classes[name]={...memo,offset:0};
  }
}
function rememberedDisplay(scope, legacyScope) {
  try {
    const key='spacemap-display:',legacy=legacyScope&&legacyScope!==scope;
    const display=JSON.parse(localStorage.getItem(key+scope)||sessionStorage.getItem(key+scope)||
      (legacy&&(localStorage.getItem(key+legacyScope)||sessionStorage.getItem(key+legacyScope)))||'null');
    // A new window restores display choices, with density off until requested.
    if(display){display.scope=scope;for(const layer of display.layers||[])layer.maps=[];}
    return display;
  } catch (_) { return null; }
}
function rememberDisplay() {
  const display=captureDisplayTransfer(state.spec);
  if(!display)return;
  state.displayTransfer=display;
  // Read only when opening this window/project. Other open windows keep their
  // own live choices; no storage listener moves another viewer or reviewer.
  try { localStorage.setItem('spacemap-display:'+display.scope,JSON.stringify(display)); } catch (_) { /* optional browser storage */ }
}
function captureDisplayTransfer(next) {
  const from=state.layers.find(l=>l.type==='image'),to=next.layers?.find(l=>l.type==='image');
  const scope=displayScope(next);
  if(!scope)return null;
  if(scope!==displayScope(state.spec)||!from||!to)return rememberedDisplay(scope,state.reviewer?null:next.session);
  if(!state.viewReady)return state.displayTransfer?.scope===scope?state.displayTransfer:null;
  const copy=value=>value===undefined?undefined:JSON.parse(JSON.stringify(value));
  const fields=['visible','opacity','mode','color_by','colormap','outline','groupColours','range','filters',
    'activeSet','autoMethod','autoScope','autoParameters','autoNative','displayAuto'];
  const previous=state.displayTransfer?.scope===scope?state.displayTransfer.layers:[];
  const layers=state.layers.filter(l=>['image','cells','annotations','shapes'].includes(l.type)&&!l.cell_training).map(layer=>{
    const peers=(previous||[]).filter(old=>old.family===displayFamily(layer)), old=peers.length===1?peers[0]:null;
    const channels=copy(layer.channels);
    if(channels)channels.push(...(old?.channels||[]).filter(c=>!channels.some(current=>current.name===c.name)));
    return {
      family:displayFamily(layer),active:layer===activeCellLayer(),type:layer.type,table:displayTable(layer),
      layout:displayLayout(layer,old?.layout),
      style:Object.fromEntries(fields.filter(k=>layer[k]!==undefined).map(k=>[k,copy(layer[k])])),
      channels,hidden:[...(layer.hidden||[])],hiddenNames:[...(layer.hiddenNames||[])],shownNames:displayAnnotationNames(layer),
      maps:state.layers.filter(m=>isCellDensityMap(m)&&!m.densityDormant&&densityCellLayer(m)===layer).map(map=>({
        rowOpen:state.cellRowsOpen?.[map.id],moreOpen:state.annotationBlocksOpen?.['density-more::'+map.id],
        ...Object.fromEntries(['name','visible','opacity','vcol','fcol','fvals','radius','weighted','cmap','cmapAuto','cellScope'].map(k=>[k,copy(map[k])]))
      }))};});
  return {scope,layerGroups:{...state.layerGroups},shownOnly:!!state.layersShownOnly,
    annotationOpacity:state.annotationOpacity,annotationMode:state.annotationMode,
    annotationMarkers:{...annotationMarkerOptions()},annotationBlocks:{...state.annotationBlocksOpen},
    reviewPolygonSort:state.reviewPolygonSort,
    cellRowsOpen:{...state.cellRowsOpen},splitMarkers:splitMarkerPreferences(splitMarkerOptions()),pixelSmoothing:state.pixelSmoothing,
    scrollTop:$('panel').scrollTop,
    layers:layers.concat((previous||[]).filter(old=>!layers.some(current=>current.family===old.family)))};
}
function applyDisplayTransfer(display) {
  if(!display)return;
  if(display.annotationOpacity!==undefined)state.annotationOpacity=display.annotationOpacity;
  if(display.annotationMode)state.annotationMode=display.annotationMode;
  if(display.annotationMarkers)state.annotationMarkers={...display.annotationMarkers};
  state.annotationBlocksOpen={...display.annotationBlocks};
  state.reviewPolygonSort=display.reviewPolygonSort||'size_down';
  state.layerGroups={...display.layerGroups};state.layersShownOnly=display.shownOnly;
  state.cellRowsOpen={...display.cellRowsOpen};state.pixelSmoothing=display.pixelSmoothing;
  Object.assign(splitMarkerOptions(),display.splitMarkers);
  for(const layer of state.layers){
    const matches=display.layers.filter(old=>old.family===displayFamily(layer));
    if(matches.length!==1)continue;
    const old=matches[0];
    applyDisplayLayout(layer,old.layout);
    if(layer.type==='cells'){
      // Colour/filter fields are applied only after the new table is linked and
      // its columns checked; image opening never waits for table statistics.
      for(const k of ['visible','opacity','mode','outline'])if(k in old.style)layer[k]=old.style[k];
      if(old.active)state.cellDataLayer=layer.id;
      layer.displayTransfer=old;continue;
    }
    if(layer.review_inline){layer.visible=old.style.visible;continue;}
    if(layer.type==='image'){
      Object.assign(layer,old.style);
      for(const channel of layer.channels||[]){
        const matches=(old.channels||[]).filter(c=>c.name===channel.name);
        if(matches.length===1){const previous=matches[0];for(const k of ['visible','color','custom','low','high','label'])if(previous[k]!==undefined)channel[k]=previous[k];}
        else channel.visible=false;
      }
    }else{
      applyAnnotationDisplayTransfer(layer,old);
    }
  }
  syncAnnotationOpacity();
}
function applyAnnotationDisplayTransfer(layer,old) {
  layer.visible=old.style.visible;layer.hidden=new Set(old.hidden);layer.hiddenNames=new Set(old.hiddenNames);
  if(old.shownNames&&layer.type==='shapes'){
    layer.hiddenNames=new Set([...namesOf(layer)].filter(n=>!old.shownNames.includes(n)));
    layer.displayNameChoice={names:old.shownNames,fingerprint:annotationDisplayFingerprint(layer)};
  }
  if(old.shownNames&&layer.type==='annotations'){
    layer.displayNameChoice={names:old.shownNames,fingerprint:annotationDisplayFingerprint(layer)};
    layer.displayNamesPending=layer.displayNameChoice;
    // Only small class metadata is needed. Image opening does not wait for
    // polygons, and annotation tiles cannot flash unrelated classes first.
    annotationLegend(layer).catch(error=>{if(state.layers.includes(layer))message('Annotation display: '+error.message,true);});
  }
}

function matchingDisplayTable(layer,old) {
  if(!old)return null;
  if(state.reviewer&&old.name==='Shared cell data'){
    const shared=(layer.tables||[]).filter(t=>t.name==='Shared cell data');
    if(shared.length===1)return shared[0].id;
  }
  if(old.combined&&old.combinedPath===dataView.combined?.path&&layer.combinedTable)return layer.combinedTable;
  const folder=file=>String(file||'').replace(/\/[^/]+$/,'');
  const candidates=(layer.tables||[]).filter(t=>{
    const file=t.source_file||t.file||'';
    if(old.file&&file===old.file)return true;
    // Only workflow-assigned per-image tables from the same folder can vary
    // by filename. User-imported data must be the exact same bound file.
    return !layer.imported_cells&&['Phenotyping results','Cell tagging results','Segmentation measurements'].includes(old.name)&&
      cellDataInfo(layer,t.id).name===old.name&&old.file&&file&&folder(file)===folder(old.file);
  });
  return candidates.length===1?candidates[0].id:null;
}
async function finishDisplayTransfer(generation) {
  await state.phenotypingLink;
  if(generation!==state.initGeneration)return;
  await Promise.all(state.layers.filter(l=>l.displayTransfer).map(async layer=>{
    const old=layer.displayTransfer;delete layer.displayTransfer;
    const table=matchingDisplayTable(layer,old.table);if(!table)return;
    const choice=state.groupPickSequence,sourceChoice=dataView.choiceSequence;
    const current=()=>generation===state.initGeneration&&state.layers.includes(layer)&&choice===state.groupPickSequence&&sourceChoice===dataView.choiceSequence;
    const before=JSON.stringify([layer.table,layer.color_by,layer.filters,[...(layer.hidden||[])],layer.range]);
    try{
      const schema=await getJSON('/cells/'+layer.token+'/columns?t='+encodeURIComponent(table));
      if(!current()||before!==JSON.stringify([layer.table,layer.color_by,layer.filters,[...(layer.hidden||[])],layer.range]))return;
      const columns=new Set(schema.columns||[]),colour=old.style.color_by||'';
      layer.table=table;
      layer.color_by=columns.has(colour)?colour:'';
      layer.filters=(old.style.filters||[]).filter(f=>columns.has(f.col)&&f.col!==(layer.table_roles?.[table]?.roi_column||layer.roi_column));
      layer.hidden=new Set(layer.color_by?old.hidden:[]);layer.range=layer.color_by?old.style.range:null;
      for(const k of ['colormap','groupColours'])if(k in old.style)layer[k]=old.style[k];
      layer.selection=null;layer.live=null;layer.viewSource='Viewer filters';invalidateCellColumns(layer);
      bump(layer,{keepVisible:true});
      if(old.maps.length)state.layers.filter(m=>isCellDensityMap(m)&&densityCellLayer(m)===layer).forEach(m=>removeLayer(m,true));
      for(const [index,saved] of old.maps.entries()){
        if(!current())break;
        if(saved.vcol&&!columns.has(saved.vcol)||saved.fcol&&!columns.has(saved.fcol))continue;
        const id='density-'+Date.now().toString(36)+'-'+index;
        cellRowsOpen()[id]=saved.rowOpen??true;
        (state.annotationBlocksOpen||={})['density-more::'+id]=saved.moreOpen??false;
        await addLayerToView({...saved,id,type:'density',cellDensity:true,added:true,
          token:layer.token,image_token:layer.image_token,cellsId:layer.id,dataMode:'viewer',table:table,tables:layer.tables},layer);
      }
      if(current()){buildPanel();saveView();}
    }catch(error){if(current())message('Cell display settings: '+error.message,true);}
  }));
}
