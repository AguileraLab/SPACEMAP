/* Bridge the authoritative Flow workspace to the existing cell layer and Data. */
'use strict';
(() => {
  let bus,channel,windowRef,docked,controller,selection,lastLayer,source='cells',lastContext='',active=false,selectionVersion=0;
  let spatialColumns={x:'',y:''},lastAppearance='';
  const sender='studio-'+Math.random().toString(36).slice(2);
  function session(){return new URLSearchParams(location.search).get('session')||state.spec?.session;}
  function connect(){const id=session();if(!id)return;if(bus&&channel===id)return;bus?.close();channel=id;
    bus=typeof BroadcastChannel!=='undefined'?new BroadcastChannel('spacemap-flow-'+id):null;if(bus)bus.onmessage=e=>receive(e.data);}
  function send(message){connect();bus?.postMessage({sender,...message});controller?.receive({sender,...message});}
  async function receive(message){if(message.sender===sender)return;active=true;
    if(message.type==='columns'&&!spatialColumns.x){const pair=[['x_centroid','y_centroid'],['centroid_x','centroid_y'],['X','Y'],['x','y']].find(([a,b])=>message.columns.includes(a)&&message.columns.includes(b));
      if(pair){[spatialColumns.x,spatialColumns.y]=pair;for(const [axis,value] of [['X',pair[0]],['Y',pair[1]]]){const input=document.querySelector('[aria-label="Flow spatial '+axis+' column"]');if(input)input.value=value;}lastContext='';}}
    if(message.type==='source-request'){source=message.source;lastContext='';lastAppearance='';publishContext();}
    if(message.type==='selection'){selection=message;await showSelection();}
    if(message.type==='clear-selection')clear();
    if(message.type==='labels'){selection=null;state.layers.filter(l=>l.type==='cells').forEach(l=>{l.flowSelection=null;bump(l);});dataView.flowSelection=null;dataView.flowSelectionName='';
      await dataRefreshCombined();if(dataView.source?.combined)await dataLoad();document.querySelectorAll('.manual-reference-card').forEach(c=>c.refreshSelection?.());}
    if(message.type==='figure'){const figures=document.getElementById('studio-figures'),dock=document.getElementById('studio-dock');if(figures&&dock){
      const card=el('div',{class:'studio-sec','data-flow-figure':''},[el('div',{class:'studio-sec-h',text:'Flow gating'}),el('img',{src:message.preview,alt:'Saved Flow gating figure',style:'width:100%;cursor:zoom-in',onclick:()=>{const opened=window.open('','_blank');if(opened){const img=opened.document.createElement('img');img.src=message.preview;img.style.maxWidth='100%';opened.document.body.append(img);}}})]);figures.append(card);dock.hidden=false;}}
  }
  async function showSelection(){if(!selection)return;const picked=selection,sequence=state.groupPickSequence=(state.groupPickSequence||0)+1;dataView.groupPickVersion=(dataView.groupPickVersion||0)+1;const version=++selectionVersion;await state.phenotypingLink;
    if(version!==selectionVersion||sequence!==state.groupPickSequence||!selection)return;consolidateCellResults();const layer=state.layers.find(l=>l.type==='cells'&&l.combinedTable&&l.id===state.cellDataLayer)||state.layers.find(l=>l.type==='cells'&&l.combinedTable);
    if(!layer||selection.table!==dataView.combined?.path){message('Flow gating needs Phenotyping results linked to this image. Check Cell data in Cells.',true);return;}
    if(layer.table!==layer.combinedTable){const table=await cellTableWith(layer,[],layer.combinedTable,false);if(version!==selectionVersion||sequence!==state.groupPickSequence||!table)return;layer.table=table;invalidateCellColumns(layer);}
    if(version!==selectionVersion||sequence!==state.groupPickSequence)return;
    layer.filters=[];layer.hidden=new Set();layer.range=null;layer.viewSource="Flow gating";layer.flowSelection=picked.selection.token;layer.flowMode=picked.mode;layer.flowPopulation=picked.selection;layer.selection=null;layer.live=null;layer.visible=true;lastLayer=layer;
    state.cellDataLayer=layer.id;dataView.filters=[];dataView.groupPick=null;dataView.onlySelected=false;dataView.selected.clear();
    dataView.flowSelection=picked.selection.token;dataView.flowSelectionName='';bump(layer);buildPanel();
    if(dataView.holder?.isConnected){if(!dataView.source?.combined)await dataChoose('combined');else await dataLoad();}
    if(version!==selectionVersion||sequence!==state.groupPickSequence)return;
    document.querySelectorAll('[data-flow-count]').forEach(n=>{const roi=dataLayerRoi(layer);n.textContent=(picked.selection.by_image[roi]||0).toLocaleString()+' cells in this image';});
  }
  function clear(options={}){selection=null;lastLayer=null;selectionVersion++;dataView.flowSelection=null;dataView.flowSelectionName='';state.layers.filter(l=>l.type==='cells').forEach(l=>{if(l.flowSelection){delete l.flowSelection;delete l.flowPopulation;delete l.flowMode;l.viewSource='All cells';bump(l);}});if(!options.silent){if(dataView.source?.combined)dataLoad();buildPanel();}}
  window.clearFlowGateView=clear;
  function context(){const layer=state.layers.find(l=>l.type==='cells'&&l.combinedTable),table=dataView.combined?.path,roi=layer?dataLayerRoi(layer):'';
    if(!layer||!table)return null;
    if(source==='cells'){const picked=state.manualReferencePick;return {kind:'cells',table,roi,ids:picked?.layer?.token===layer.token?picked.labels||[]:[]};}
    if(source==='image'){const filters=(layer.filters||[]).slice();const column=layer.table_roles?.[layer.combinedTable]?.roi_column||dataView.info?.roi_column;
      if(column)filters.push({col:column,values:[roi]});return filters.length?{kind:'filters',table,filters}:{kind:'image',table,roi};}
    if(source==='filters'){if(!dataView.source?.combined||!dataView.queryReady)return null;const body=dataView.resultBody||dataBody({});const filters=(body.filters||[]).slice();
      if(body.rows?.this_image&&body.rows.column)filters.push({col:body.rows.column,values:[body.rows.value]});return {kind:'filters',table,filters};}
    if(source==='region'||source==='viewport'){let geometries=[];
      if(source==='region'){const picked=state.picks;if(picked?.layer?.features)geometries=picked.layer.features.filter(f=>picked.ids.has(f.properties.id)).map(f=>f.geometry);}
      else {const item=baseItem();if(item&&state.viewer){const b=item.viewportToImageRectangle(state.viewer.viewport.getBounds(true));geometries=[{type:'Polygon',coordinates:[[[b.x,b.y],[b.x+b.width,b.y],[b.x+b.width,b.y+b.height],[b.x,b.y+b.height],[b.x,b.y]]]}];}}
      return {kind:'region',table,roi,geometries,x:spatialColumns.x,y:spatialColumns.y,join:'or'};}
    return null;
  }
  function publishContext(){if(!active)return;const appearance=state.clusterReview||null,appearanceStamp=JSON.stringify(appearance);
    if(appearanceStamp!==lastAppearance){lastAppearance=appearanceStamp;send({type:'appearance',appearance});}
    const next=context(),stamp=JSON.stringify([source,next]);if(stamp===lastContext)return;lastContext=stamp;send({type:'context',source,context:next});}
  const timer=setInterval(()=>{if(!active)return;connect();publishContext();if(selection&&lastLayer&&!state.layers.includes(lastLayer))showSelection().catch(e=>message(e.message,true));},750);
  window.addEventListener('beforeunload',()=>{clearInterval(timer);bus?.close();});
  window.launchFlowWorkspace=holder=>{
    connect();active=true;
    const note=el('p',{class:'muted','data-flow-count':'',text:'Linked to the image and Data.'});
    const open=()=>{const script=[...document.scripts].find(s=>s.src.includes('/flow-gating.js'));
      const url=new URL('flow-gating.html',script?.src||location.href);url.searchParams.set('session',channel);url.searchParams.set('api',new URL(api('/'),location.href).href.replace(/\/$/,''));
      windowRef=window.open(url.href,'spacemap-flow-'+channel,'popup,width=1250,height=900');if(windowRef){windowRef.focus();note.textContent='Flow window opened.';}else note.textContent='Popup blocked. Use Open window or Dock below.';};
    const dock=async()=>{if(docked){docked.hidden=false;return;}docked=el('section',{class:'flow-docked','aria-label':'Gates workbench'});docked.style.cssText='top:52%;left:22%;right:3%;bottom:2%;';
      const close=el('button',{class:'btn',text:'Hide gates',onclick:()=>docked.hidden=true,style:'position:absolute;right:6px;top:6px;z-index:10'}),mount=el('div',{style:'height:100%'});docked.append(close,mount);document.body.append(docked);
      controller=await SPACEMAPFlow.mount(mount,{channel,get:getJSON,post:postJSON,plotly:api('/flow_plotly.js'),message:receive});lastContext='';publishContext();};
    const coordinate=el('details',{},[el('summary',{text:'Spatial coordinates'})]);
    const x=el('input',{placeholder:'X centroid column','aria-label':'Flow spatial X column',value:spatialColumns.x}),y=el('input',{placeholder:'Y centroid column','aria-label':'Flow spatial Y column',value:spatialColumns.y});
    x.onchange=()=>{spatialColumns.x=x.value;lastContext='';};y.onchange=()=>{spatialColumns.y=y.value;lastContext='';};coordinate.append(x,y,help('Used for Live region or viewport selection. Coordinates must be in full image pixels; cells are included by their centroid.'));
    holder.append(el('div',{class:'studio-sec'},[el('div',{class:'row'},[el('button',{class:'btn primary',text:'Open window',onclick:open}),el('button',{class:'btn',text:'Dock below',onclick:dock})]),note,
      el('div',{class:'row'},[el('button',{class:'btn',text:'Show Data',onclick:async()=>{showDataTab();await dataChoose('combined');}}),el('button',{class:'btn',text:'Clear gate view',onclick:clear}),el('button',{class:'btn',text:'Show figures',onclick:()=>{const d=document.getElementById('studio-dock');if(d)d.hidden=false;}})]),coordinate]));
    open();
  };
})();
