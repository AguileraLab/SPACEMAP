/* Shared polygon selection: exact IDs, tiled borders, no geometry import on split. */
'use strict';

function annotationSelection(layer) {
  const selections=state.annotationSelections||(state.annotationSelections=new Map());
  if(!selections.has(layer))selections.set(layer,new Set());
  return selections.get(layer);
}
function componentQuery(layer, values={}) {
  const route=annotationComponentRoute(layer);
  return route+(route.includes('?')?'&':'?')+new URLSearchParams(values);
}
function annotationSelectedGroups() {
  return [...(state.annotationSelections||[])].filter(([layer,ids])=>ids.size&&state.layers.includes(layer));
}
function clearAnnotationSelection() {
  state.areaSelectionEpoch=(state.areaSelectionEpoch||0)+1;
  for(const [layer,ids] of state.annotationSelections||[]) {ids.clear();annotationOutline(layer);}
  state.picks=null;state.selected=null;
  syncAnnotationSelection();
}
function selectAnnotations(layer, ids, {replace=false, toggle=false}={}) {
  state.areaSelectionEpoch=(state.areaSelectionEpoch||0)+1;
  if(replace)for(const [other,selection] of state.annotationSelections||[])selection.clear();
  const selected=annotationSelection(layer);
  const linked=new Map(),sources=new Map(state.layers.filter(l=>l.token).map(l=>[l.token,l]));
  for(const preview of state.layers.filter(l=>l.review_inline))for(const f of preview.features||[]) {
    const p=f.properties;let id,partner;
    if(layer===preview){const source=sources.get(p.review_source);if(source){id=p.id;partner=[source,p.review_polygon];rememberAnnotationOutline(source,{...f,properties:{...p,id:p.review_polygon}});}}
    else if(layer.token&&p.review_source===layer.token){id=p.review_polygon;partner=[preview,p.id];}
    if(partner){if(!linked.has(id))linked.set(id,[]);linked.get(id).push(partner);}
  }
  for(const value of ids) {
    const id=String(value),partners=linked.get(id)||[];
    // Review previews and their source polygon are one selection, even when the
    // source family is hidden. Match both source and ID, never a label alone.
    const on=!(toggle&&(selected.has(id)||partners.some(([l,k])=>annotationSelection(l).has(k))));
    on?selected.add(id):selected.delete(id);
    partners.forEach(([l,k])=>on?annotationSelection(l).add(k):annotationSelection(l).delete(k));
  }
  state.picks={layer,ids:selected};
  state.selected=null;
  for(const [source] of state.annotationSelections||[])annotationOutline(source);
  syncAnnotationSelection();
}

function areaAnnotationLayers(){
  return state.layers.filter(l=>l.visible&&l.opacity!==0&&(l.type==='annotations'||l.type==='shapes'&&!l.training_part&&!l.cell_training));
}
function areaSelectionAvailable(){
  // Keep the target switch available even when the chosen target is hidden.
  return areaAnnotationLayers().length>0||!!cellLayer();
}
function cellAreaView(layer){
  return {t:layer.table||'',f:JSON.stringify(layer.filters||[]),col:layer.color_by||'',hide:[...(layer.hidden||[])].join('\x1f'),
    lo:String(layer.range?.[0]??''),hi:String(layer.range?.[1]??''),selection:layer.selection||'',
    flow:layer.flowSelection||'',flow_mode:layer.flowMode||'filter',live:layer.live?'1':'0'};
}
function clearCellAreaOutline(){
  const overlay=state.cellAreaOverlay;
  if(overlay?.item&&overlay.viewer.world.getIndexOfItem(overlay.item)>=0)overlay.viewer.world.removeItem(overlay.item);
  state.cellAreaOverlay=null;
}
function cellAreaOutline(layer,token){
  clearCellAreaOutline();const viewer=state.viewer,source=tileSource(layer);
  const overlay=state.cellAreaOverlay={viewer,layer,item:null};
  source.getTileUrl=(level,x,y)=>api('/cells/'+layer.token+'/tile/'+(layer.info.max_zoom-level)+'/'+x+'/'+y+'.png?'+new URLSearchParams({t:layer.table,selection:token,mode:'outline',ol:'ffffff'}));
  viewer.addTiledImage({tileSource:source,success:event=>{
    if(state.cellAreaOverlay!==overlay||state.viewer!==viewer||!layer.visible){viewer.world.removeItem(event.item);return;}
    overlay.item=event.item;
  }});
}
async function selectVisibleArea(points,additive){
  const epoch=state.areaSelectionEpoch=(state.areaSelectionEpoch||0)+1,generation=state.initGeneration,xy=points.map(p=>[p.x,p.y]);
  const current=()=>epoch===state.areaSelectionEpoch&&generation===state.initGeneration;
  try{
    if(state.areaSelectTarget==='cells'){
      const layer=cellLayer();if(!layer||layer.opacity===0)return message('Show a cell layer first.',true);
      const view=cellAreaView(layer),previous=state.cellInspectionSelection;
      const payload={action:'select_cells',token:layer.token,view,points:xy,add:additive&&previous?.layer===layer&&previous.table===layer.table?[...previous.members.keys()]:[]};
      const result=await postJSON(state.reviewer?'/select_cells':reviewRoute(),payload);
      if(!current()||!state.layers.includes(layer)||!layer.visible||layer.opacity===0||JSON.stringify(view)!==JSON.stringify(cellAreaView(layer)))return;
      clearCellInspection({notify:false});
      state.cellInspectionSelection={layer,table:layer.table,members:new Map(result.ids.map(label=>[label,{label}]))};
      if(result.count)cellAreaOutline(layer,result.selection);
      document.dispatchEvent(new CustomEvent('spacemap-cells-picked',{detail:{layer,labels:result.ids}}));
      message(result.count.toLocaleString()+' cells selected.');return;
    }
    const layers=areaAnnotationLayers(),features=layers.map(l=>l.features),snapshot=()=>JSON.stringify(areaAnnotationLayers().map(l=>[l.id,l.token,[...(l.hidden||[])],[...(l.hiddenNames||[])],state.versions[l.id]])),signature=snapshot();
    if(!layers.length)return message('Show annotation polygons first.',true);
    const matches=await Promise.all(layers.map(async layer=>{
      if(layer.type==='annotations')return [layer,await postJSON(annotationComponentRoute(layer),{action:'area',points:xy,hidden:[...(layer.hidden||[])]})];
      const xs=points.map(p=>p.x),ys=points.map(p=>p.y),bounds=[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];
      const features=(layer.features||[]).filter(f=>featureShown(layer,f)).filter(f=>{const b=featureBounds(f);return b.x0>=bounds[0]&&b.y0>=bounds[1]&&b.x1<=bounds[2]&&b.y1<=bounds[3];});
      if(layer.review_inline&&state.reviewer){
        if(!features.length)return [layer,{ids:[]}];
        return [layer,await postJSON('/select_review_polygons',{items:[...new Set(features.map(f=>f.properties.review_item))],focus:features[0].properties.review_item,points:xy})];
      }
      return [layer,await postJSON(reviewRoute(),{action:'select_shapes',features,points:xy})];
    }));
    if(!current()||signature!==snapshot()||layers.some((l,i)=>l.features!==features[i]))return;
    if(!additive)clearAnnotationSelection();
    for(const [layer,result] of matches)if(result.ids.length)selectAnnotations(layer,result.ids);
    message(annotationSelectedGroups().reduce((n,[,ids])=>n+ids.size,0).toLocaleString()+' polygons selected.');
  }catch(error){if(current())message(error.message,true);}
}
function syncAnnotationSelection() {
  renderShapes();
  document.dispatchEvent(new CustomEvent('spacemap-annotations-picked', {detail:{groups:annotationSelectedGroups()}}));
  // The left Annotation panel has its own list and edit actions.
  refreshRegionList();
}
function removeAnnotationOutline(layer) {
  layer.annotationOutlineEpoch=(layer.annotationOutlineEpoch||0)+1;
  cancelPendingAnnotationOutline(layer);
  const overlay=layer.annotationSelectionOverlay;
  if(overlay?.item&&overlay.viewer.world.getIndexOfItem(overlay.item)>=0)overlay.viewer.world.removeItem(overlay.item);
  layer.annotationOutlineSvg?.remove();layer.annotationOutlineSvg=null;
  layer.annotationSelectionOverlay=null;
  layer.annotationOutlineSignature=null;
}
function cancelPendingAnnotationOutline(layer){
  const pending=layer.annotationOutlinePending;
  if(pending?.item&&pending.viewer.world.getIndexOfItem(pending.item)>=0)pending.viewer.world.removeItem(pending.item);
  layer.annotationOutlinePending=null;
}
function rememberAnnotationOutline(layer,feature){
  if(layer.type!=='annotations')return;
  const cache=layer.annotationOutlineCache||(layer.annotationOutlineCache=new Map()),revision=feature.componentRevision;
  if(revision&&layer.annotationGeometryRevision!==revision){cache.clear();layer.annotationGeometryRevision=revision;}
  const id=String(feature.properties.id),d=pathOf(feature.geometry);
  cache.delete(id);cache.set(id,{d,name:feature.properties.classification?.name||feature.properties.name});
  // Exact geometry for recent clicks only; large selections stay tiled.
  let size=[...cache.values()].reduce((n,row)=>n+row.d.length,0);
  while(cache.size>64||size>2000000){const first=cache.keys().next().value;size-=cache.get(first).d.length;cache.delete(first);}
}
function drawAnnotationOutline(layer,ids){
  const cache=layer.annotationOutlineCache,parts=[];let complete=ids.size<=64;
  for(const id of ids){const row=cache?.get(id);if(!row){complete=false;continue;}if(!layer.hidden?.has(row.name))parts.push(row.d);}
  let group=layer.annotationOutlineSvg;
  if(!parts.length){group?.remove();layer.annotationOutlineSvg=null;return complete;}
  if(!group){
    group=document.createElementNS(SVG_NS,'g');group.setAttribute('data-annotation-selection',layer.id);
    for(const [colour,width] of [['#000',6],['#fff',2]]){
      const path=document.createElementNS(SVG_NS,'path');
      for(const [key,value] of Object.entries({fill:'none',stroke:colour,'stroke-width':width,'vector-effect':'non-scaling-stroke','stroke-linejoin':'round'}))path.setAttribute(key,value);
      group.append(path);
    }
    layer.annotationOutlineSvg=group;
  }
  for(const path of group.children)shapeAttribute(path,'d',parts.join(''));
  if(group.parentNode!==state.svgRoot)state.svgRoot?.append(group);
  return complete;
}
async function annotationOutline(layer) {
  if(layer.type!=='annotations')return;
  const ids=annotationSelection(layer), viewer=state.viewer, generation=state.initGeneration;
  const signature=JSON.stringify([[...ids].sort(),[...(layer.hidden||[])].sort(),layer.token]);
  if(ids.size&&layer.visible&&layer.opacity!==0&&signature===layer.annotationOutlineSignature){
    if(layer.annotationOutlinePending)drawAnnotationOutline(layer,ids);
    return;
  }
  const epoch=layer.annotationOutlineEpoch=(layer.annotationOutlineEpoch||0)+1;
  if(!ids.size||!layer.visible||layer.opacity===0||!state.layers.includes(layer)){removeAnnotationOutline(layer);return;}
  cancelPendingAnnotationOutline(layer);
  layer.annotationOutlineSignature=signature;
  const complete=drawAnnotationOutline(layer,ids),previous=layer.annotationSelectionOverlay;
  if(complete){
    if(previous?.item&&previous.viewer.world.getIndexOfItem(previous.item)>=0)previous.viewer.world.removeItem(previous.item);
    layer.annotationSelectionOverlay=null;return;
  }
  // Keep already drawn borders for additive selections. Clear obsolete borders
  // on a deliberate deselection; they must never look selected while loading.
  if(previous&&(!previous.ids||[...previous.ids].some(id=>!ids.has(id))||previous.hidden!==JSON.stringify([...(layer.hidden||[])].sort()))){
    if(previous.viewer.world.getIndexOfItem(previous.item)>=0)previous.viewer.world.removeItem(previous.item);
    layer.annotationSelectionOverlay=null;
  }
  try {
    const found=await postJSON(annotationComponentRoute(layer),{action:'outline',ids:[...ids]});
    if(epoch!==layer.annotationOutlineEpoch||viewer!==state.viewer||generation!==state.initGeneration||!layer.visible||!state.layers.includes(layer))return;
    const source=tileSource(layer), hidden=[...(layer.hidden||[])].join('\x1f');
    source.getTileUrl=(level,x,y)=>api(componentQuery(layer,{selection:found.selection,z:layer.info.max_zoom-level,x,y,hide:hidden}));
    viewer.addTiledImage({tileSource:source,opacity:0,preload:true,blendTime:0,success:event=>{
      if(epoch!==layer.annotationOutlineEpoch||viewer!==state.viewer||!layer.visible||!state.layers.includes(layer)){
        if(viewer.world.getIndexOfItem(event.item)>=0)viewer.world.removeItem(event.item);return;
      }
      const pending=layer.annotationOutlinePending={viewer,item:event.item};
      const swap=()=>{
        if(layer.annotationOutlinePending!==pending||epoch!==layer.annotationOutlineEpoch||viewer!==state.viewer||!layer.visible||!state.layers.includes(layer))return;
        const old=layer.annotationSelectionOverlay;
        event.item.setOpacity(1);
        if(old?.item&&viewer.world.getIndexOfItem(old.item)>=0)viewer.world.removeItem(old.item);
        layer.annotationSelectionOverlay={viewer,item:event.item,ids:new Set(ids),hidden:JSON.stringify([...(layer.hidden||[])].sort())};
        layer.annotationOutlinePending=null;
        layer.annotationOutlineSvg?.remove();layer.annotationOutlineSvg=null;
      };
      // TiledImage success means its metadata exists, not that its pixels exist.
      if(event.item.getFullyLoaded())setTimeout(swap,0);
      else event.item.addHandler('fully-loaded-change',event=>{if(event.fullyLoaded)setTimeout(swap,0);});
    }});
  }catch(error){if(epoch===layer.annotationOutlineEpoch&&viewer===state.viewer){layer.annotationOutlineSignature=null;message(error.message,true);}}
}

async function annotationHit(point) {
  let best=null;
  // Smallest visible polygon wins; hidden classes and holes cannot intercept a click.
  for(const layer of state.layers.filter(l=>l.visible&&l.opacity!==0&&(l.type==='annotations'||l.type==='shapes'&&!l.training_part&&!l.cell_training))){
    if(layer.type==='shapes'){
      for(const feature of layer.features||[])if(featureShown(layer,feature)&&contains(feature,point.x,point.y)){
        const area=featureArea(feature);if(!best||area<best.area||area===best.area&&layer.review_inline)best={layer,feature,area};
      }
    }else{
      const found=await getJSON(componentQuery(layer,{x:point.x,y:point.y,hide:[...(layer.hidden||[])].join('\x1f')}));
      if(found.feature){const feature={...found.feature,componentRevision:found.revision};rememberAnnotationOutline(layer,feature);const area=featureArea(feature);if(!best||area<best.area)best={layer,feature,area};}
    }
  }
  return best;
}
function pickAnnotation(point, additive=false) {
  if(!additive)state.annotationPickEpoch=(state.annotationPickEpoch||0)+1;
  const epoch=state.annotationPickEpoch||0,generation=state.initGeneration;
  const run=(state.annotationPickQueue||Promise.resolve()).catch(()=>{}).then(async()=>{
    if(epoch!==(state.annotationPickEpoch||0)||generation!==state.initGeneration)return true;
    const hit=await annotationHit(point);
    if(epoch!==(state.annotationPickEpoch||0)||generation!==state.initGeneration)return true;
    if(!hit){if(!additive)clearAnnotationSelection();return false;}
    if(!hit.layer.visible||!state.layers.includes(hit.layer)||hit.layer.hidden?.has(hit.feature.properties.classification?.name)||!featureShown(hit.layer,hit.feature))return true;
    selectAnnotations(hit.layer,[hit.feature.properties.id],{replace:!additive,toggle:additive});
    document.dispatchEvent(new CustomEvent('spacemap-annotation-focused',{detail:hit}));
    if(!state.reviewer)showRegionCard(hit.layer,hit.feature);
    return true;
  });
  state.annotationPickQueue=run;return run;
}

async function flagSelectedAnnotations() {
  const sources=unflaggedAnnotationSources();
  if(!sources.length)return message('The selected polygons are already in Review, or need a saved annotation source.');
  return flagReview({sources,title:sources.reduce((n,source)=>n+source.ids.length,0)+' selected polygons'});
}

function unflaggedAnnotationSources(){
  const known=new Set(state.layers.filter(l=>l.review_inline).flatMap(l=>(l.features||[]).map(f=>JSON.stringify([f.properties.review_source,String(f.properties.review_polygon)]))));
  return annotationSelectedGroups().filter(([layer])=>layer.token&&!layer.review_inline).map(([layer,ids])=>({token:layer.token,
    ids:[...ids].filter(id=>!known.has(JSON.stringify([layer.token,String(id)])))})).filter(source=>source.ids.length);
}

async function replaceAnnotationSource(layer, source) {
  const layout=displayLayout(layer),hidden=new Set(layer.hidden||[]),visible=layer.visible;
  const selected=[...annotationSelection(layer)];
  removeLayer(layer);
  const copy=await addLayerToView({...source,open:layer.open,visible,hidden:[...hidden]});
  if(copy){
    copy.hidden=hidden;copy.visible=visible;applyDisplayLayout(copy,layout);
    refreshLayer(copy);selectAnnotations(copy,selected);buildPanel();saveView();
  }
  return copy;
}
async function renameSelectedAnnotations(chosenName) {
  if(state.reviewer)return;
  if(state.annotationRenameBusy)return;
  const groups=annotationSelectedGroups().filter(([layer])=>!layer.review_inline).map(([layer,ids])=>[layer,[...ids]]);
  if(!groups.length)return message('Select polygons first.',true);
  const name=typeof chosenName==='string'?chosenName:prompt('Annotation name (an existing or new name):','');if(!name?.trim())return;
  const generation=state.initGeneration,count=groups.reduce((n,[,ids])=>n+ids.length,0);
  state.annotationRenameBusy=true;message('Saving '+count+' polygons as '+name.trim()+'…');
  try{
  for(const [layer,ids] of groups){
      if(generation!==state.initGeneration||!state.layers.includes(layer))return;
      if(layer.type==='shapes'){
        if(!layer.editable)throw Error('Make an editable copy of this source first.');
        nameRegions(layer,new Set(ids),name.trim());
      }else{
        const route=annotationComponentRoute(layer), version=await getJSON(route);
        const found=await postJSON(route,{action:'rename',ids:[...ids],name:name.trim(),revision:version.revision});
        if(generation===state.initGeneration&&state.layers.includes(layer)){
          await replaceAnnotationSource(layer,found.layer);
        }
      }
  }
  if(generation===state.initGeneration){buildPanel();refreshRegionList();message(count+' polygons labelled '+name.trim()+'.');}
  }catch(error){message(error.message,true);}
  finally{state.annotationRenameBusy=false;}
}

async function annotationLabelMenu(actions,menu,act,generation){
  const layers=state.layers.filter(l=>l.type==='annotations'||l.type==='shapes'&&!l.review_inline);
  const labels=new Set(layers.filter(l=>l.type==='shapes').flatMap(l=>[...namesOf(l)]));
  await Promise.all(layers.filter(l=>l.type==='annotations').map(async layer=>{
    try{(await annotationLegend(layer)).items.forEach(item=>labels.add(item.value));}catch(_){}
  }));
  if(!menu.isConnected||generation!==state.initGeneration)return;
  const choices=el('div',{class:'annotation-label-choices',style:'max-height:220px;overflow-y:auto'});
  for(const name of [...labels].sort((a,b)=>a.localeCompare(b)))choices.append(act('Use “'+name+'”',()=>renameSelectedAnnotations(name)));
  actions.append(choices,act('Another annotation…',()=>renameSelectedAnnotations()));menu.reposition?.();
}

async function annotationContextItems(menu,point,act) {
  if(!(state.studio||state.reviewer)||!point)return;
  const generation=state.initGeneration;
  try{
    const hit=await annotationHit(point);
    if(!hit||!menu.isConnected||generation!==state.initGeneration)return;
    const id=String(hit.feature.properties.id);
    if(!annotationSelection(hit.layer).has(id))selectAnnotations(hit.layer,[id],{replace:true});
    const actions=el('div',{class:'annotation-context-actions'});
    if(!state.reviewer){
      const count=unflaggedAnnotationSources().reduce((n,source)=>n+source.ids.length,0);
      actions.append(act(count?'Flag '+count+' new polygon'+(count===1?'':'s')+' for review':'Selected polygons already in Review',flagSelectedAnnotations,!count));
    }
    if(state.reviewer||hit.layer.review_inline||selectedReviewItems(hit).length){
      menu.prepend(actions);
      await reviewPolygonMenu(actions,hit,menu,act);
    }else {
      actions.append(act('Rename selected polygons…',()=>renameSelectedAnnotations()));
      menu.prepend(actions);
      await annotationLabelMenu(actions,menu,act,generation);
    }
    if(!actions.isConnected)menu.prepend(actions);
    menu.reposition?.();
  }catch(error){if(menu.isConnected)message(error.message,true);}
}

document.addEventListener('spacemap-annotations-picked',()=>{
  // Layers stays synchronized even when the left Review tab is not mounted.
  const reviewPicks=new Set(annotationSelectedGroups().filter(([layer])=>layer.review_inline).flatMap(([,ids])=>[...ids]));
  document.querySelectorAll('[data-review-feature]').forEach(row=>{
    const on=reviewPicks.has(row.dataset.reviewFeature);row.classList.toggle('on',on);
    const tick=row.querySelector('input[type=checkbox]');if(tick)tick.checked=on;
  });
  for(const [layer,ids] of annotationSelectedGroups()){
    for(const host of document.querySelectorAll('[data-component-source]'))if(host.dataset.componentSource===layer.token){
      host.querySelectorAll('[data-component-id]').forEach(row=>{
        const on=ids.has(row.dataset.componentId);row.classList.toggle('on',on);
        const input=row.querySelector('input[type=checkbox]');if(input)input.checked=on;
      });
      host.querySelectorAll('[data-selection-count]').forEach(node=>node.textContent=ids.size+' selected');
    }
  }
  // Cleared sources still have rendered rows.
  document.querySelectorAll('[data-component-source]').forEach(host=>{
    const layer=state.layers.find(l=>l.token===host.dataset.componentSource);if(!layer||annotationSelection(layer).size)return;
    host.querySelectorAll('[data-component-id]').forEach(row=>{row.classList.remove('on');row.querySelector('input[type=checkbox]').checked=false;});
    host.querySelectorAll('[data-selection-count]').forEach(node=>node.textContent='0 selected');
  });
});

document.addEventListener('spacemap-annotation-focused',event=>{
  for(const host of document.querySelectorAll('[data-component-source]'))if(host.dataset.componentSource===event.detail.layer.token)
    host.revealComponent?.(event.detail.feature);
});
