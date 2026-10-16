/* Individual marker crops share the main viewer's pixels, colours and display ranges.
 * Only visible cards are rendered. A bounded decoded-tile cache and four workers
 * keep cursor movement from launching a separate image viewer for every marker. */
"use strict";

const splitMarkers = {point:null, cursor:null, pinned:false, selection:null, epoch:0, cards:[], cache:new Map(), limits:new Map(), pending:new Map(), active:0, queue:[], wanted:new Set()};
const SPLIT_DEFAULTS = {open:false, all:false, lockOnCell:true, zoom:2, size:156, height:300, floating:false, width:720, x:80, y:100, tab:'markers'};

function splitMarkerPreferences(saved) {
  const source = saved || {}, out = {...SPLIT_DEFAULTS};
  for (const key of ['open','all','floating','lockOnCell']) if (typeof source[key] === 'boolean') out[key] = source[key];
  if (['markers','data'].includes(source.tab)) out.tab = source.tab;
  for (const [key,lo,hi] of [['zoom',.25,8],['size',96,280],['height',140,900],['width',320,1800],['x',0,10000],['y',0,10000]]) {
    if (Number.isFinite(source[key])) out[key] = Math.min(hi,Math.max(lo,source[key]));
  }
  return out;
}

function splitMarkerOptions() {
  if (!state.splitMarkerOptions) {
    let saved; try { saved=JSON.parse(localStorage.getItem('spacemap-split-markers')); } catch (_) { /* defaults */ }
    state.splitMarkerOptions=splitMarkerPreferences(saved);
  }
  return state.splitMarkerOptions;
}

function saveSplitMarkerOptions() {
  try { localStorage.setItem('spacemap-split-markers',JSON.stringify(splitMarkerOptions())); } catch (_) { /* private mode */ }
  saveView();
}

function splitMarkerImage() {
  const token = state.inspectedCell?.layer.image_token;
  return state.layers.find(l=>l.type==='image' && !l.info?.rgb && l.channels?.length && l.id!=='he' && (!token || l.token===token)) ||
    state.layers.find(l=>l.type==='image' && !l.info?.rgb && l.channels?.length && l.id!=='he');
}

function splitMarkerChannels(layer) {
  return (layer?.channels || []).filter(c=>splitMarkerOptions().all || c.visible);
}

function resetSplitMarkers() {
  stopSplitMarkerPreview();splitMarkers.generation=(splitMarkers.generation||0)+1;
  splitMarkers.epoch++; splitMarkers.point=null; splitMarkers.cursor=null; splitMarkers.selection=null; splitMarkers.pinned=false;
  splitMarkers.lastInspection=null;splitMarkers.limits.clear();
  splitMarkers.limitWait=null;splitMarkers.limitError=null;splitMarkers.failures?.clear();
  splitMarkers.signature=null;
  for (const bitmap of splitMarkers.cache.values()) bitmap.close();
  splitMarkers.cache.clear();
  if (splitMarkers.grid) splitMarkers.grid.replaceChildren();
  splitMarkers.cards=[];
}

function splitMarkerTarget(result) {
  const points=(result.members || [result]).map(m=>m.point || {x:Number(m.details?.x_centroid),y:Number(m.details?.y_centroid)})
    .filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
  if (!points.length) return null;
  return {point:{x:points.reduce((v,p)=>v+p.x,0)/points.length,y:points.reduce((v,p)=>v+p.y,0)/points.length},
    span:points.length>1?Math.max(Math.max(...points.map(p=>p.x))-Math.min(...points.map(p=>p.x)),Math.max(...points.map(p=>p.y))-Math.min(...points.map(p=>p.y))):0};
}

function splitMarkerSelection(layer, result) {
  if (state.inspectedCell?.result!==result || splitMarkers.lastInspection===result) return;
  splitMarkers.lastInspection=result;
  if(!splitMarkerOptions().lockOnCell)return;
  const target=splitMarkerTarget(result);if(!target)return;
  splitMarkers.selection=result;
  splitMarkers.point=target.point;
  splitMarkers.pinned=true;
  splitMarkers.span=target.span;
  syncSplitMarkers();
}

function releaseSplitMarkerSelection(forget=true) {
  splitMarkers.pinned=false;splitMarkers.selection=null;splitMarkers.span=0;
  if(forget)splitMarkers.lastInspection=null;
  if(splitMarkers.cursor)splitMarkers.point={...splitMarkers.cursor};
  if(splitMarkers.panel)syncSplitMarkers();
}

function followSplitMarkerCursor(position) {
  const layer=splitMarkerImage(), point=layer?.item?.viewerElementToImageCoordinates(position) || toImage(position);
  if (!point || !layer) return;
  splitMarkers.cursor={x:point.x,y:point.y};
  if(splitMarkers.pinned&&(!state.inspectedCell||!state.layers.includes(state.inspectedCell.layer)))releaseSplitMarkerSelection();
  if(splitMarkers.pinned)return;
  splitMarkers.point={...splitMarkers.cursor}; splitMarkers.span=0;
  if(splitMarkerOptions().open)scheduleSplitMarkers();
}

function ensureSplitMarkers() {
  if (splitMarkers.panel) return;
  const opt=splitMarkerOptions();
  const toggle=el('button',{id:'split-marker-toggle',type:'button',text:state.studio?'▴ Marker tiles · Data':'▴ Marker tiles',title:'Show individual markers around the cursor or selected cells','aria-controls':'split-marker-panel',onclick:()=>{
    opt.open=!opt.open; if(!opt.open)opt.floating=false; syncSplitMarkers();saveSplitMarkerOptions();
  }});
  const scope=select([['selected','Selected markers'],['all','All markers']],opt.all?'all':'selected',value=>{
    opt.all=value==='all';syncSplitMarkers();saveSplitMarkerOptions();
  }); scope.setAttribute('aria-label','Marker tile scope');
  const zoom=select([.25,.5,1,2,4,8].map(v=>[String(v),v===1?'100% (native)':(v*100)+'%']),String(opt.zoom),value=>{
    opt.zoom=Number(value);scheduleSplitMarkers();saveSplitMarkerOptions();
  }); zoom.setAttribute('aria-label','Marker tile zoom');zoom.title='Zoom inside every tile; right-click a tile for the same choices.';
  const size=el('input',{type:'range',min:96,max:280,step:4,value:opt.size,'aria-label':'Marker tile size',title:'Make tiles smaller to fit more markers',oninput:()=>{
    opt.size=Number(size.value);syncSplitMarkers();
  },onchange:saveSplitMarkerOptions});
  const follow=el('button',{class:'btn small',text:'Follow cursor',title:'Resume following the cursor; clicking a cell pins the location',onclick:()=>{
    releaseSplitMarkerSelection(false);
  }});
  const lock=el('input',{type:'checkbox','aria-label':'Lock marker tiles on cell selection',onchange:()=>{
    opt.lockOnCell=lock.checked;
    if(!opt.lockOnCell)releaseSplitMarkerSelection(false);
    else if(state.inspectedCell){splitMarkers.lastInspection=null;splitMarkerSelection(state.inspectedCell.layer,state.inspectedCell.result);}
    saveSplitMarkerOptions();
  }});
  const lockLabel=el('label',{class:'inline-check',title:'Keep tiles on selected cells. Deselect or close the cell card to follow the pointer again.'},[lock,el('span',{text:'Lock on cell'})]);
  const float=el('button',{class:'btn small',text:'Float',title:'Float above Studio; drag the header to move',onclick:()=>{
    opt.floating=!opt.floating;syncSplitMarkers();saveSplitMarkerOptions();
  }});
  const hide=el('button',{class:'btn small',text:'Hide ▾',title:'Collapse to the bottom of the viewer',onclick:()=>{
    opt.open=false;opt.floating=false;syncSplitMarkers();saveSplitMarkerOptions();
  }});
  // Tabs: Marker tiles, and (in Studio) Data, the cells' table linked to the image.
  const tabs=el('div',{class:'split-tabs',role:'tablist'});
  const tab=(id,text)=>el('button',{type:'button',class:'split-tab',role:'tab','data-tab':id,text,onclick:()=>{opt.tab=id;syncSplitMarkers();saveSplitMarkerOptions();}});
  tabs.append(tab('markers','Marker tiles'));
  if(state.studio)tabs.append(tab('data','Data'));
  const markerTools=el('span',{class:'split-marker-tools'},[scope,zoom,el('label',{class:'split-size'},[el('span',{text:'Size'}),size]),lockLabel,follow]);
  const head=el('div',{class:'split-marker-head'},[tabs,markerTools,el('span',{class:'grow'}),float,hide]);
  const note=el('div',{class:'split-marker-note',role:'status'});
  const grid=el('div',{class:'split-marker-grid'});
  const resize=el('div',{class:'split-marker-resize',role:'separator',tabindex:0,'aria-label':'Resize marker panel',title:'Drag up or down to resize. Arrow keys also resize.'});
  const corner=el('div',{class:'split-marker-corner',title:'Drag to resize the floating panel'});
  const dataBox=el('div',{class:'split-data'});
  const panel=el('section',{id:'split-marker-panel','aria-label':'Split marker viewer'},[resize,head,note,grid,dataBox,corner]);
  $('stage').append(toggle,panel);
  Object.assign(splitMarkers,{toggle,panel,grid,note,head,scope,zoom,size,follow,lock,float,tabs,markerTools,dataBox});
  grid.addEventListener('scroll',()=>scheduleSplitMarkers(),{passive:true});
  panel.addEventListener('contextmenu',event=>{
    event.preventDefault();event.stopPropagation();closeContextMenu();
    const menu=el('div',{id:'ctxmenu',class:'split-marker-menu',role:'menu'},[el('div',{class:'ctx-title',text:'Marker tile zoom'}),
      el('div',{class:'ctx-row'},[.25,.5,1,2,4,8].map(v=>el('button',{text:v===1?'100% (native)':v*100+'%',onclick:()=>{
        opt.zoom=v;zoom.value=String(v);closeContextMenu();scheduleSplitMarkers();saveSplitMarkerOptions();
      }})))]);
    menu.style.left=Math.max(8,Math.min(event.clientX,innerWidth-360))+'px';
    document.body.append(menu);menu.style.top=Math.max(8,Math.min(event.clientY,innerHeight-menu.offsetHeight-8))+'px';
  });
  const drag=(handle,kind)=>handle.addEventListener('pointerdown',event=>{
    if(event.button!==0 || (kind==='move' && (!opt.floating || event.target.closest('button,input,select,label'))))return;
    event.preventDefault();handle.setPointerCapture(event.pointerId);
    const start={x:event.clientX,y:event.clientY,height:panel.offsetHeight,width:panel.offsetWidth,left:panel.offsetLeft,top:panel.offsetTop};
    const move=e=>{
      if(kind==='move'){opt.x=start.left+e.clientX-start.x;opt.y=start.top+e.clientY-start.y;}
      else if(kind==='corner'){opt.width=Math.max(320,start.width+e.clientX-start.x);opt.height=Math.max(140,start.height+e.clientY-start.y);}
      else opt.height=Math.max(140,start.height+start.y-e.clientY);
      placeSplitMarkers();scheduleSplitMarkers();
    };
    const end=()=>{handle.removeEventListener('pointermove',move);handle.removeEventListener('pointerup',end);handle.removeEventListener('pointercancel',end);saveSplitMarkerOptions();};
    handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',end);handle.addEventListener('pointercancel',end);
  });
  drag(head,'move');drag(resize,'height');drag(corner,'corner');
  resize.addEventListener('keydown',e=>{if(['ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();e.stopPropagation();opt.height+=e.key==='ArrowUp'?24:-24;opt.height=Math.max(140,opt.height);placeSplitMarkers();scheduleSplitMarkers();saveSplitMarkerOptions();}});
  splitMarkers.observer=new ResizeObserver(()=>{placeSplitMarkers();scheduleSplitMarkers();});
  splitMarkers.observer.observe($('stage'));
  window.addEventListener('resize',()=>{placeSplitMarkers();scheduleSplitMarkers();});
}

function placeSplitMarkers() {
  const {panel,toggle}=splitMarkers;if(!panel)return;
  const opt=splitMarkerOptions(), stage=$('stage');
  // Floating panels are body children so they can cross both Studio sidebars.
  const parent=opt.floating&&opt.open?document.body:stage;if(panel.parentElement!==parent)parent.append(panel);
  panel.classList.toggle('floating',opt.floating&&opt.open);panel.classList.toggle('is-open',opt.open);
  panel.inert=!opt.open;panel.setAttribute('aria-hidden',String(!opt.open));
  toggle.setAttribute('aria-expanded',String(opt.open));toggle.hidden=opt.open;
  const height=Math.min(opt.height,Math.max(140,(opt.floating?innerHeight:stage.clientHeight)-65));
  const dockedHeight=opt.open&&!opt.floating?height:0;
  stage.style.setProperty('--marker-dock-height',dockedHeight+'px');
  if(state.viewer?.navigator?.element)state.viewer.navigator.element.style.bottom=dockedHeight+'px';
  panel.style.height=height+'px';panel.style.setProperty('--marker-size',opt.size+'px');
  if(opt.floating){
    const width=Math.min(opt.width,innerWidth-16);panel.style.width=width+'px';
    opt.x=Math.max(8,Math.min(opt.x,innerWidth-width-8));opt.y=Math.max(8,Math.min(opt.y,innerHeight-height-8));
    panel.style.left=opt.x+'px';panel.style.top=opt.y+'px';
  }else{panel.style.width='';panel.style.left='';panel.style.top='';}
}

function syncSplitMarkers() {
  ensureSplitMarkers();
  const opt=splitMarkerOptions(),image=splitMarkerImage();
  placeSplitMarkers();splitMarkers.toggle.disabled=!image && !state.studio;
  // The Data tab (Studio): the cells' table instead of the marker tiles, in the same panel.
  const dataTab=!!state.studio&&opt.tab==='data';
  splitMarkers.tabs.querySelectorAll('.split-tab').forEach(b=>{const on=b.dataset.tab===(dataTab?'data':'markers');b.classList.toggle('on',on);b.setAttribute('aria-selected',String(on));});
  splitMarkers.markerTools.hidden=dataTab;splitMarkers.grid.hidden=dataTab;splitMarkers.note.hidden=dataTab;splitMarkers.dataBox.hidden=!dataTab;
  if(dataTab){if(opt.open)dataViewInto(splitMarkers.dataBox);stopSplitMarkerPreview();return;}
  splitMarkers.scope.value=opt.all?'all':'selected';splitMarkers.zoom.value=String(opt.zoom);splitMarkers.size.value=opt.size;
  splitMarkers.follow.disabled=!splitMarkers.pinned;splitMarkers.float.textContent=opt.floating?'Dock':'Float';
  splitMarkers.lock.checked=opt.lockOnCell;
  splitMarkers.float.setAttribute('aria-label',opt.floating?'Dock marker tiles':'Float marker tiles');
  const channels=splitMarkerChannels(image), signature=JSON.stringify([image?.token,channels.map(c=>[c.index,c.label,c.color,c.low,c.high]),opt.size]);
  if(signature!==splitMarkers.signature){
    splitMarkers.signature=signature;
    splitMarkers.cards=channels.map(channel=>{
      const canvas=el('canvas',{'aria-label':(channel.label||channel.name)+' marker crop'});
      const card=el('div',{class:'split-marker-tile','data-marker':channel.name},[
        el('div',{class:'split-marker-label',title:channel.label||channel.name},[el('span',{class:'split-marker-swatch',style:'background:'+channel.color}),el('span',{text:channel.label||channel.name})]),canvas]);
      return {channel,canvas,card};
    });splitMarkers.grid.replaceChildren(...splitMarkers.cards.map(c=>c.card));
  }
  splitMarkers.note.textContent=!image?'Open a multichannel image.':!channels.length?'Select markers in Layers, or choose All markers.':
    (splitMarkers.pinned?(splitMarkers.selection?.members?splitMarkers.selection.members.length+' cells':'Cell '+splitMarkers.selection?.label)+' · pinned':'Follow cursor')+
    ' · '+channels.length+' markers';
  if(!opt.open){stopSplitMarkerPreview();return;}
  if(!splitMarkers.point && state.viewer && image?.item){
    const p=image.item.viewportToImageCoordinates(state.viewer.viewport.getCenter(true));splitMarkers.point={x:p.x,y:p.y};
  }
  scheduleSplitMarkers();
}

function scheduleSplitMarkers(changed=true) {
  if(!splitMarkerOptions().open || (state.studio&&splitMarkerOptions().tab==='data'))return;
  if(changed)splitMarkers.epoch++;
  if(splitMarkers.timer)return;
  splitMarkers.timer=requestAnimationFrame(()=>{splitMarkers.timer=null;renderSplitMarkers();});
}

function stopSplitMarkerPreview() {
  splitMarkers.epoch++;cancelAnimationFrame(splitMarkers.timer);splitMarkers.timer=null;
  splitMarkers.wanted=new Set();pruneSplitMarkerRequests();splitMarkers.rendering=false;
}

function splitMarkerGeometry(image,point,size,zoom,span=0,dpr=1) {
  const effective=Math.min(zoom,span>0?size/(span+32):zoom), width=size/effective;
  const z=Math.min(image.info.max_zoom,Math.max(0,Math.floor(Math.log2(1/(effective*dpr)))));
  return {x:point.x-width/2,y:point.y-width/2,width,z,scale:2**z};
}

function cachedSplitMarkerBitmap(url) {
  const cache=splitMarkers.cache;
  const bitmap=cache.get(url);
  if(bitmap){cache.delete(url);cache.set(url,bitmap);}
  return bitmap;
}

function pruneSplitMarkerRequests() {
  for(const [url,job] of splitMarkers.pending){
    // An AI contact sheet may share a download with the preview. Cursor
    // movement must not cancel that independent consumer.
    if(job.persistent || splitMarkers.wanted.has(url))continue;
    job.controller.abort();splitMarkers.pending.delete(url);
    if(!job.started)job.reject(new DOMException('Preview moved','AbortError'));
  }
  splitMarkers.queue=splitMarkers.queue.filter(job=>!job.controller.signal.aborted);
  for(const url of splitMarkers.failures?.keys()||[])if(!splitMarkers.wanted.has(url))splitMarkers.failures.delete(url);
}

function splitMarkerBitmap(url,{preview=false}={}) {
  const bitmap=cachedSplitMarkerBitmap(url);if(bitmap)return Promise.resolve(bitmap);
  let job=splitMarkers.pending.get(url);
  if(job){if(!preview)job.persistent=true;if(preview)job.preview=true;return job.promise;}
  job={url,preview,persistent:!preview,controller:new AbortController(),generation:splitMarkers.generation};
  job.promise=new Promise((resolve,reject)=>Object.assign(job,{resolve,reject}));
  splitMarkers.pending.set(url,job);splitMarkers.queue.push(job);pumpSplitMarkerRequests();
  return job.promise;
}

function pumpSplitMarkerRequests() {
  while(splitMarkers.active<4&&splitMarkers.queue.length){
    const job=splitMarkers.queue.shift();if(job.controller.signal.aborted)continue;
    job.started=true;splitMarkers.active++;
    (async()=>{
      try{
        const response=await fetch(job.url,{credentials:'same-origin',signal:job.controller.signal});
        if(!response.ok)throw Error('Marker tile: HTTP '+response.status);
        const bitmap=await createImageBitmap(await response.blob());
        if(job.controller.signal.aborted || job.generation!==splitMarkers.generation){bitmap.close();throw new DOMException('Preview moved','AbortError');}
        const cache=splitMarkers.cache;cache.set(job.url,bitmap);
        // Enough for four source tiles per marker in a fifty-marker panel.
        while(cache.size>256){const key=cache.keys().next().value;cache.get(key).close();cache.delete(key);}
        job.resolve(bitmap);
      }catch(error){
        if(job.preview&&error.name!=='AbortError'&&splitMarkers.wanted.has(job.url)){
          (splitMarkers.failures ||= new Map()).set(job.url,{epoch:splitMarkers.epoch,error});
        }
        job.reject(error);
      }finally{
        if(splitMarkers.pending.get(job.url)===job)splitMarkers.pending.delete(job.url);
        splitMarkers.active--;pumpSplitMarkerRequests();
        if(job.preview&&!job.controller.signal.aborted&&job.generation===splitMarkers.generation)scheduleSplitMarkers(false);
      }
    })();
  }
}

function applySplitMarkerLimits(image,styles) {
  for(const c of styles){const kept=splitMarkers.limits.get(image.token+':'+image.info.fingerprint+':'+c.index);
    if(kept){if(c.low===undefined)c.low=kept.low;if(c.high===undefined)c.high=kept.high;}}
}

async function splitMarkerLimits(image,styles) {
  const generation=splitMarkers.generation;
  const key=c=>image.token+':'+image.info.fingerprint+':'+c.index;
  applySplitMarkerLimits(image,styles);
  const missing=styles.filter(c=>c.low===undefined||c.high===undefined);
  await fillLimits(image,missing);
  if(generation===splitMarkers.generation)missing.forEach(c=>splitMarkers.limits.set(key(c),{low:c.low,high:c.high}));
}

function markerCropTiles(image,channel,{x,y,width,z,scale}) {
  const span=image.info.tile_size*scale,tiles=[],source={...image,checker:undefined,channels:[{...channel,visible:true}]};
  for(let ty=Math.max(0,Math.floor(y/span));ty<=Math.floor(Math.min(y+width,image.info.height-1)/span);ty++){
    for(let tx=Math.max(0,Math.floor(x/span));tx<=Math.floor(Math.min(x+width,image.info.width-1)/span);tx++){
      tiles.push({tx,ty,url:tileUrl(source,z,tx,ty,0,true)});
    }
  }
  return tiles;
}

function drawMarkerCrop(canvas,image,geometry,tiles,bitmaps,{dpr=1,pixels=splitMarkerOptions().size,current=()=>true,rings=[]}={}) {
  if(!current())return false;
  const size=Math.round(pixels*dpr), {x,y,width,scale}=geometry;
  // Paint privately, then swap the entire crop so a slow request cannot mix positions.
  const scratch=canvas.splitScratch ||= document.createElement('canvas');
  if(scratch.width!==size||scratch.height!==size){scratch.width=size;scratch.height=size;}
  const ctx=scratch.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,size,size);
  ctx.imageSmoothingEnabled=width>size || !!state.pixelSmoothing;
  const factor=size/width;
  // Stitch on the pyramid's integer pixel grid, then resample once. This avoids
  // hairline seams when a crop crosses a tile boundary at fractional zoom.
  const mosaic=canvas.splitMosaic ||= document.createElement('canvas'),ox=Math.floor(x/scale),oy=Math.floor(y/scale),side=Math.ceil(width/scale)+2;
  if(mosaic.width!==side||mosaic.height!==side){mosaic.width=side;mosaic.height=side;}
  const mosaicCtx=mosaic.getContext('2d');mosaicCtx.fillStyle='#000';mosaicCtx.fillRect(0,0,mosaic.width,mosaic.height);
  tiles.forEach(({tx,ty},i)=>mosaicCtx.drawImage(bitmaps[i],tx*image.info.tile_size-ox,ty*image.info.tile_size-oy));
  ctx.drawImage(mosaic,x/scale-ox,y/scale-oy,width/scale,width/scale,0,0,size,size);
  ctx.strokeStyle='#fff';ctx.lineWidth=dpr;ctx.shadowColor='#000';ctx.shadowBlur=2*dpr;
  if(rings.length)for(const ring of rings){ctx.beginPath();ring.forEach(([px,py],i)=>ctx[i?'lineTo':'moveTo']((px-x)*factor,(py-y)*factor));ctx.closePath();ctx.stroke();}
  else {const mid=size/2;ctx.beginPath();ctx.moveTo(mid-4*dpr,mid);ctx.lineTo(mid+4*dpr,mid);ctx.moveTo(mid,mid-4*dpr);ctx.lineTo(mid,mid+4*dpr);ctx.stroke();}
  ctx.shadowBlur=0;
  const mpp=image.info.pixel_size_um;
  if(mpp){const length=niceLength(width*mpp/5),px=length/mpp*factor;
    ctx.fillStyle='#fff';ctx.fillRect(8*dpr,size-18*dpr,px,2*dpr);ctx.font=10*dpr+'px sans-serif';ctx.fillText(length+' µm',8*dpr,size-4*dpr);}
  if(!current())return false;
  if(canvas.width!==size||canvas.height!==size){canvas.width=size;canvas.height=size;}
  canvas.getContext('2d').drawImage(scratch,0,0);return true;
}

async function paintMarkerCrop(canvas,image,channel,geometry,options={}) {
  const tiles=markerCropTiles(image,channel,geometry),bitmaps=[];
  for(const tile of tiles){
    if(options.current&&!options.current())return false;
    bitmaps.push(await splitMarkerBitmap(tile.url));
  }
  return drawMarkerCrop(canvas,image,geometry,tiles,bitmaps,options);
}

function renderSplitMarkers() {
  const opt=splitMarkerOptions(),image=splitMarkerImage(),point=splitMarkers.point;
  if(!opt.open||(state.studio&&opt.tab==='data')||!image||!point){stopSplitMarkerPreview();return;}
  const epoch=splitMarkers.epoch,current=()=>opt.open&&epoch===splitMarkers.epoch&&state.layers.includes(image);
  const rect=splitMarkers.grid.getBoundingClientRect(),cards=splitMarkers.cards.filter(({card})=>{const r=card.getBoundingClientRect();return r.bottom>rect.top&&r.top<rect.bottom;});
  const pixels=cards[0]?.canvas.clientWidth || opt.size;
  const dpr=Math.min(2,window.devicePixelRatio||1),geo=splitMarkerGeometry(image,point,pixels,opt.zoom,splitMarkers.pinned?splitMarkers.span:0,dpr);
  const rings=splitMarkers.pinned&&state.inspectedCell?.result===splitMarkers.selection?state.cellOutline||[]:[];
  let failures=0;splitMarkers.rendering=false;
  try {
    const styles=cards.map(c=>({...c.channel}));applySplitMarkerLimits(image,styles);
    if(styles.some(c=>c.low===undefined||c.high===undefined)){
      splitMarkers.wanted=new Set();pruneSplitMarkerRequests();
      const key=JSON.stringify([image.token,styles.map(c=>[c.index,c.low,c.high])]);
      if(splitMarkers.limitError?.key===key&&splitMarkers.limitError.epoch===epoch)throw splitMarkers.limitError.error;
      splitMarkers.rendering=true;
      if(splitMarkers.limitWait!==key){
        splitMarkers.limitWait=key;
        splitMarkerLimits(image,styles).catch(error=>{if(splitMarkers.limitWait===key)splitMarkers.limitError={key,epoch:splitMarkers.epoch,error};})
          .finally(()=>{if(splitMarkers.limitWait===key){splitMarkers.limitWait=null;scheduleSplitMarkers(false);}});
      }
      return;
    }
    const plans=cards.map((entry,i)=>({entry,tiles:markerCropTiles(image,styles[i],geo)}));
    splitMarkers.wanted=new Set(plans.flatMap(plan=>plan.tiles.map(tile=>tile.url)));pruneSplitMarkerRequests();
    // Cached crops draw together on this animation frame. Downloads only warm
    // the cache and request a redraw at the *latest* pointer position.
    for(const {entry,tiles} of plans){
      const failed=tiles.map(tile=>splitMarkers.failures?.get(tile.url)).find(f=>f?.epoch===epoch);
      if(failed){failures++;entry.card.classList.remove('loading');entry.card.setAttribute('aria-busy','false');entry.card.classList.add('failed');entry.card.title=failed.error.message;continue;}
      const bitmaps=tiles.map(tile=>cachedSplitMarkerBitmap(tile.url));
      if(bitmaps.every(Boolean)){
        if(entry.card.dataset.rendered!==String(epoch)){
          drawMarkerCrop(entry.canvas,image,geo,tiles,bitmaps,{dpr,pixels,current,rings});
          entry.card.classList.remove('failed','loading');entry.card.setAttribute('aria-busy','false');entry.card.title='';entry.card.dataset.rendered=String(epoch);
          entry.card.dataset.center=point.x+','+point.y;
        }
      }else{
        splitMarkers.rendering=true;
        entry.card.classList.add('loading');entry.card.setAttribute('aria-busy','true');entry.card.title='Loading current position…';
        tiles.forEach((tile,i)=>{if(!bitmaps[i])splitMarkerBitmap(tile.url,{preview:true}).catch(()=>{});});
      }
    }
    if(failures)splitMarkers.note.textContent=failures+' marker tiles could not load. Move the cursor or reopen the panel to retry.';
  }catch(error){splitMarkers.rendering=false;if(current())splitMarkers.note.textContent=error.message;}
}

// The AI receives labelled contact sheets from the same crop renderer, never a
// screenshot of an unrelated cursor location. Measurements go as structured data.
async function askCellAI(layer,result,button) {
  if(result.aiBusy)return;
  const image=state.layers.find(l=>l.type==='image'&&l.token===layer.image_token&&l.id!=='he');
  if(!image||!state.studio?.askCell) {message('Open Studio to ask the Assistant about a cell.',true);return;}
  if(result.profile?.status!=='ready'){message('Wait for the cell measurements to load first.',true);return;}
  result.aiBusy=true;button.disabled=true;button.textContent='Preparing…';
  const profile=result.profile,table=layer.table;
  const current=()=>state.inspectedCell?.result===result&&result.profile===profile&&layer.table===table&&state.layers.includes(layer);
  try {
    const target=splitMarkerTarget(result);if(!target)throw Error('This cell has no image position available. Click it in the viewer again.');
    const point=target.point,opt={...splitMarkerOptions()},span=target.span;
    const channels=splitMarkerChannels(image).map(c=>({...c}));
    if(!channels.length)throw Error('Select at least one image marker, or choose All markers in Marker tiles.');
    await splitMarkerLimits(image,channels);
    let rings=[];
    const members=result.members||[result];
    for(const member of members.slice(0,100)){
      if(!current())return;
      const outline=member.rings||(await getJSON('/cells/'+layer.token+'/outline?label='+member.label+'&t='+encodeURIComponent(layer.table||''))).rings||[];
      rings.push(...outline);
    }
    const geo=splitMarkerGeometry(image,point,opt.size,opt.zoom,span);
    const images=[],perPage=24,side=180,cols=6;
    for(let offset=0;offset<channels.length;offset+=perPage){
      const page=channels.slice(offset,offset+perPage),sheet=document.createElement('canvas');
      sheet.width=cols*side;sheet.height=Math.ceil(page.length/cols)*(side+24)+28;
      const ctx=sheet.getContext('2d');ctx.fillStyle='#11141b';ctx.fillRect(0,0,sheet.width,sheet.height);
      ctx.fillStyle='#fff';ctx.font='13px sans-serif';ctx.fillText((result.members?members.length+' selected cells':'Cell '+result.label)+' · '+(offset+1)+'–'+(offset+page.length)+' of '+channels.length+' markers',8,19);
      let next=0;
      await Promise.all(Array.from({length:Math.min(4,page.length)},async()=>{while(next<page.length&&current()){
        const i=next++,canvas=document.createElement('canvas');
        if(!await paintMarkerCrop(canvas,image,page[i],geo,{pixels:side,current,rings}))return;
        const x=(i%cols)*side,y=Math.floor(i/cols)*(side+24)+28;
        ctx.drawImage(canvas,x,y,side,side);ctx.fillStyle=page[i].color;ctx.fillRect(x+5,y+side+6,10,10);
        ctx.fillStyle='#fff';ctx.fillText(page[i].label||page[i].name,x+20,y+side+16,side-24);
      }}));
      if(!current())return;
      images.push(sheet.toDataURL('image/png'));
      if(images.length===4)break;
    }
    const columns=[...new Set([result.suggestions?.provenance?.column,layer.learnColumn,layer.color_by,state.mainPhenotype,
      ...Object.keys(result.details||{}).filter(k=>/^phenotype_/i.test(k))].filter(Boolean))];
    const known=new Set((result.suggestions?.probabilities||result.suggestions?.choices||[]).map(p=>p.phenotype));
    for(const col of columns.slice(0,3)){
      try {const found=await getJSON('/cells/'+layer.token+'/legend?t='+encodeURIComponent(layer.table||'')+'&col='+encodeURIComponent(col));
        (found.items||[]).forEach(i=>known.add(String(i.value)));}catch(_){ /* retain the model's vocabulary */ }
    }
    const finite=v=>Number.isFinite(v)?Number(v.toPrecision(6)):null;
    const names=new Set((result.profile.markers||[]).map(r=>r.marker));
    const numeric=[...new Set(members.flatMap(m=>Object.keys(m.details||{})))].filter(k=>!names.has(k)&&members.some(m=>Number.isFinite(m.details?.[k]))).slice(0,120);
    const data={cell_ids:members.slice(0,100).map(m=>m.label),total_cells:members.length,image:state.spec.title,
      source:layer.profileSource||'table',transforms:result.profile.steps||[],aggregation:result.members?'Mean after transforming each cell; mixed phenotypes remain possible.':'Single cell',
      measurements:(result.profile.markers||[]).map(r=>({marker:r.marker,raw:finite(r.mean),signal:finite(r.signal),transformed:finite(r.view)})),
      morphology_and_other_features:Object.fromEntries(numeric.map(k=>{const vals=members.map(m=>m.details?.[k]).filter(Number.isFinite);return[k,finite(vals.reduce((a,b)=>a+b,0)/vals.length)];})),
      current_labels:result.details,model:result.suggestions?.provenance,predictions:result.suggestions?.probabilities||result.suggestions?.choices,
      dataset_phenotypes:[...known].slice(0,120),crop:{center:point,width_pixels:geo.width,markers:channels.slice(0,96).map(c=>({name:c.name,colour:c.color,low:c.low,high:c.high})),outline:'White outline marks the selected cell(s).'},
      purpose:'Review phenotype evidence only; do not assign labels or run an action.'};
    // Keep full measurements separate from duplicate details, within chat's view budget.
    data.current_labels=Object.fromEntries(Object.entries(result.details||{}).filter(([k,v])=>/^phenotype_|tissue|tumor|annotation|region/i.test(k)&&typeof v!=='object').slice(0,60));
    if(!current())return;
    await state.studio.askCell({images,context:data,text:'Suggest the most plausible phenotype for '+(result.members?'these selected cells':'this selected cell')+
      ' using the supplied marker measurements, morphology, labelled marker tiles and existing phenotype vocabulary. Compare plausible alternatives and explain contradictory evidence or uncertainty. '+
      'Distinguish measured values from display brightness; do not infer positivity solely from tile colours. Propose a name for review only; do not change labels or run a command.'});
  }catch(error){message(error.message,true);}
  finally{result.aiBusy=false;if(state.inspectedCell?.result===result)showCellCard(layer,result);}
}

// Open the panel on its Data tab (a table opened from elsewhere).
function showDataTab() {
  const opt=splitMarkerOptions();opt.open=true;opt.tab='data';syncSplitMarkers();saveSplitMarkerOptions();
}
