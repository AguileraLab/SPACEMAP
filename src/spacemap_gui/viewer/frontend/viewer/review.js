/* One review queue; the image viewer, marker controls and layer renderer are shared. */
'use strict';

const REVIEW_NAMES = {annotation:'Annotations', cells:'Cell labels', leiden:'Leiden', som:'SOM', vortex:'Vortex',
  subcluster:'Subclusters', neighbourhood:'Neighbourhoods', tls:'TLS', gland:'Glands', niche:'Niches',
  gland_type:'Gland types', niche_type:'Niche types'};
const REVIEW_STATES = {requested:'Reviewer nomination', pending:'Needs review', answered:'Decision received', unresolved:'Unable to decide',
  applying:'Applying…', applied:'Applied', conflict:'Changed — check again'};

function reviewRoute() { return '/studio/' + encodeURIComponent(state.spec?.session || new URLSearchParams(location.search).get('session')) + '/review'; }

function ensureImageReviewPolygons(){
  if(!state.studio&&!state.reviewer)return;
  const image=state.layers.find(l=>l.type==='image');if(!image)return;
  const key=image.token+'|'+state.initGeneration;
  if(state.reviewPolygonLoad?.key===key)return;
  const request=state.reviewPolygonLoad={key};
  const route=state.reviewer?'/image_preview?image=':reviewRoute()+'?action=image_preview&image=';
  getJSON(route+encodeURIComponent(image.token)).then(found=>{
    if(state.reviewPolygonLoad!==request||!state.layers.includes(image))return;
    const signature=JSON.stringify([image.token,found.revision]);
    let layer=state.layers.find(l=>l.id==='review-focus');
    if(layer?.reviewPreviewKey&&JSON.parse(layer.reviewPreviewKey)[1]>=found.revision||!layer&&!found.features.length)return;
    if(!layer){layer={id:'review-focus',type:'shapes',name:'Review polygons',review_inline:true,editable:false,visible:false,opacity:1,regions:true,features:[]};state.layers.push(layer);}
    layer.features=found.features;layer.reviewPreviewKey=signature;
    const ids=new Set(found.features.map(f=>f.properties.id)),picks=annotationSelection(layer);
    for(const id of picks)if(!ids.has(id))picks.delete(id);
    const sources=new Map(state.layers.filter(l=>l.token).map(l=>[l.token,l]));
    for(const f of found.features){const p=f.properties,source=sources.get(p.review_source);if(source&&annotationSelection(source).has(p.review_polygon))picks.add(p.id);}
    renderShapes();buildPanel();
  }).catch(error=>{if(state.reviewPolygonLoad===request){request.error=error.message;message('Review polygons: '+error.message,true);}});
}
document.addEventListener('spacemap-review-changed',()=>{state.reviewPolygonLoad=null;if(state.layerGroups?.annotations)ensureImageReviewPolygons();});

function forgetReviewPreviews(ids) {
  if(!ids?.length)return;
  const removed=new Set(ids);let changed=false;
  // Only remove the review overlay. Original polygons and their selection stay.
  const image=state.layers.find(l=>l.type==='image');
  state.reviewPolygonLoad=image?{key:image.token+'|'+state.initGeneration}:null;
  for(const layer of state.layers.filter(l=>l.review_inline)){
    const picks=annotationSelection(layer),kept=[];
    for(const feature of layer.features||[]){
      if(removed.has(feature.properties.review_item)){picks.delete(feature.properties.id);changed=true;}
      else kept.push(feature);
    }
    layer.features=kept;
  }
  if(changed){renderShapes();buildPanel();}
}

async function removeReviewItems(ids) {
  if(state.reviewer)return;
  ids=[...new Set(ids||[])];
  if(!ids.length)return message('Select review items first.',true);
  try{
    const found=await postJSON(reviewRoute(),{action:'remove',items:ids});
    forgetReviewPreviews(found.removed);
    document.dispatchEvent(new CustomEvent('spacemap-review-changed',{detail:found}));
    message(found.removed.length+' removed from review. Original polygons and saved edits are kept.');
  }catch(error){message(error.message,true);}
}

async function flagReview(payload) {
  if (state.reviewer) return;
  try {
    const found=await postJSON(reviewRoute(), {...payload,action:payload.action||'flag',cell_context:reviewCellContext()});
    state.lastFlaggedReview=found.items.map(item=>item.id);
    document.dispatchEvent(new CustomEvent('spacemap-review-changed'));
    message(found.items.length+' item(s) flagged for review.');
    $('message').append(' ',el('button',{class:'btn small',text:'Suggestions…',onclick:()=>reviewSuggestionDialog(state.lastFlaggedReview)}));
    return found;
  } catch(error) {message(error.message,true);}
}

async function reviewSuggestionDialog(ids, {images=[],kind='annotation'}={}) {
  ids=ids||[];
  if(state.reviewer||!ids.length&&!images.length)return message('Select review items first.',true);
  const dialog=el('dialog',{class:'review-dialog'}),status=el('div',{role:'status',text:'Loading choices…'});
  dialog.append(el('div',{class:'row'},[el('h3',{class:'grow',text:'Review suggestions'}),el('button',{class:'btn tiny',text:'×','aria-label':'Close suggestions',onclick:()=>dialog.close()})]),status);
  dialog.addEventListener('close',()=>dialog.remove());document.body.append(dialog);dialog.showModal();
  const scope=select([...(ids.length?[['items','Selected items ('+ids.length+')']]:[]),...(images.length?[['images','Selected images ('+images.length+')']]:[])],ids.length?'items':'images',()=>load());
  scope.setAttribute('aria-label','Suggestions apply to');
  scope.title='Selected images includes all pending flags of the chosen review type, across every page. Existing decisions stay unchanged.';
  const type=select(Object.entries(REVIEW_NAMES),kind||'annotation',()=>load());type.setAttribute('aria-label','Suggestions review type');
  const scopeRow=el('div',{class:'row'},[el('label',{text:'Apply to'}),scope,type]);
  const body=el('div');if(images.length)dialog.append(scopeRow);dialog.append(body);let epoch=0;
  async function load(){
  const request=++epoch;type.hidden=scope.value!=='images';status.textContent='Loading choices…';body.replaceChildren();
  try {
    const found=await postJSON(reviewRoute(),{action:'suggestion_options',...(scope.value==='images'?{images,kind:type.value}:{ids})});
    if(!dialog.isConnected||request!==epoch)return;
    const items=found.items,names=new Set(found.options),checked=new Set(items[0]?.suggestions||[]);
    // For a mixed batch, check only choices common to every selected item.
    for(const name of checked)if(!items.every(item=>item.suggestions.includes(name)))checked.delete(name);
    if(items.every(item=>item.kind==='annotation')){
      await Promise.all(state.layers.filter(layer=>layer.type==='annotations'||layer.type==='shapes'&&!layer.cell_training).map(async layer=>{
        if(layer.type==='annotations'){try{(await annotationLegend(layer)).items?.forEach(row=>names.add(row.value));}catch(_){} }
        else namesOf(layer).forEach(name=>names.add(name));
      }));
    }
    if(!dialog.isConnected||request!==epoch)return;
    const search=el('input',{type:'search',placeholder:'Find choices…','aria-label':'Find suggested choices'}),options=el('div',{class:'review-suggestion-options'});
    const all=el('input',{type:'checkbox','aria-label':'Select matching choices'});
    const render=()=>{
      const visible=[...names].sort((a,b)=>a.localeCompare(b)).filter(name=>name.toLowerCase().includes(search.value.toLowerCase()));
      all.checked=!!visible.length&&visible.every(name=>checked.has(name));all.indeterminate=visible.some(name=>checked.has(name))&&!all.checked;
      options.replaceChildren(...visible.map(name=>{
        const tick=el('input',{type:'checkbox','aria-label':'Suggest '+name});tick.checked=checked.has(name);
        tick.onchange=()=>{tick.checked?checked.add(name):checked.delete(name);render();};
        return el('label',{class:'inline'},[tick,el('span',{text:name})]);
      }));
    };
    search.oninput=render;all.onchange=()=>{for(const name of names)if(name.toLowerCase().includes(search.value.toLowerCase()))all.checked?checked.add(name):checked.delete(name);render();};
    const custom=el('input',{maxlength:200,placeholder:'Add another choice…','aria-label':'Custom suggestion'});
    const add=()=>{const name=custom.value.trim();if(name){names.add(name);checked.add(name);custom.value='';search.value='';render();}};
    custom.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();add();}};
    const remember=el('input',{type:'checkbox','aria-label':'Use these choices for future flags'});remember.checked=true;
    const note=el('textarea',{rows:2,placeholder:'Optional note','aria-label':'Review note'});let noteChanged=false;
    if(items.every(item=>item.note===items[0]?.note))note.value=items[0]?.note||'';
    note.oninput=()=>{noteChanged=true;};
    const save=el('button',{class:'btn primary',disabled:items.length?null:'',text:'Apply to '+items.length+' item'+(items.length===1?'':'s'),onclick:async()=>{
      save.disabled=true;try{
        await postJSON(reviewRoute(),{action:'suggestions',items:Object.fromEntries(items.map(item=>[item.id,item.version])),choices:[...checked],remember:remember.checked,...(noteChanged?{note:note.value}:{})});
        dialog.close();document.dispatchEvent(new CustomEvent('spacemap-review-changed'));message('Review suggestions saved.');
      }catch(error){status.textContent=error.message;save.disabled=false;}
    }});
    status.textContent=scope.value==='images'?(found.skipped?found.skipped+' with decisions or other states left unchanged.':items.length?'':'No pending flags of this type in the selected images.'):'';
    body.append(el('div',{class:'row'},[all,search]),options,el('div',{class:'row'},[custom,el('button',{class:'btn small',text:'Add',onclick:add})]),
      note,el('div',{class:'row'},[el('label',{class:'inline'},[remember,el('span',{text:'Use for future flags'})]),
        help('Remember these choices for the same annotation name or result column in this project. Flag for review then adds them without opening a dialog. You can change suggestions later in Review. Other is an optional choice; reviewers can also enter their own label or choose Unable to decide.')]),
      el('div',{class:'row'},[save,el('button',{class:'btn small',text:'Cancel',onclick:()=>dialog.close()})]));render();search.focus();
  }catch(error){if(request===epoch)status.textContent=error.message;}
  }
  load();
}

function flagAnnotationReview(layer, ids) {
  if (!ids.length) return message('Select polygons first.', true);
  if (layer.preview?.saving || layer.savePromise) return message('Wait for the polygon edit to save first.', true);
  flagReview({token:layer.token, ids:Array.from(ids, String), title:layer.name + ' · ' + ids.length + ' polygons'});
}

async function showReviewItem(found, current=()=>true) {
  const spec=found.spec, image=spec.layers.find(l=>l.type==='image');
  const same=state.layers.some(l=>l.type==='image' && l.token===image.token);
  if (!same) {
    if(state.reviewer){
      // A shared reviewer has a private view, never the owner's command session.
      dataView.loadSequence=(dataView.loadSequence||0)+1;dataView.choiceSequence=(dataView.choiceSequence||0)+1;dataView.source=null;dataView.sourceKey=null;dataView.info=null;dataView.result=null;dataView.grid?.replaceChildren();
      await init(spec);
    }else{
      // Keep the owner's normal image identity and registered data layers. A
      // local review-only init leaves the server on the old image, causing the
      // next Studio command to fail its current-view check.
      const session=state.spec?.session||new URLSearchParams(location.search).get('session');
      const path=found.item?.target?.image;
      if(!session||!path)throw new Error('Reopen this review item from the project.');
      const added=await postJSON('/session/'+encodeURIComponent(session)+'/add-image',{path});
      if(!current())return false;
      const entry=added.images.find(item=>item.id===added.added);
      if(!entry)throw new Error('The review image is no longer available in the project.');
      const result=await runCommand('open_image',{image:entry.group+' | '+entry.name});
      if(result.status!=='done')throw new Error(result.message||'Could not open the review image.');
    }
  }
  if(!current()||!state.layers.some(l=>l.type==='image'&&l.token===image.token))return false;
  const generation=state.initGeneration;
  // Add review context to the owner's full image view, without replacing it.
  // For a newly opened reviewer view init already loaded these layers.
  if(same||!state.reviewer){
    const resultIds=new Set(spec.layers.filter(l=>l.type==='cells'&&l.id.startsWith('review-result-')).map(l=>l.id));
    for(const layer of [...state.layers])if(layer.id==='review-focus'||(layer.id.startsWith('review-result-')&&!resultIds.has(layer.id)))removeLayer(layer,true);
    for(const layer of spec.layers.filter(l=>l.type!=='image')){
      if(!current()||generation!==state.initGeneration)return false;
      const existing=state.layers.find(l=>l.token&&l.token===layer.token);
      // A remembered source may predate the current review item. Refresh only
      // its scoped binding; keep the reviewer's own display choices.
      if(existing&&state.reviewer&&layer.review_item)existing.review_item=layer.review_item;
      if(!existing)await addLayerToView(layer);
    }
  }
  if(!current()||generation!==state.initGeneration)return false;
  const focus=state.layers.find(l=>l.id==='review-focus');
  if(focus?.features?.[0]) {
    const source=state.layers.find(l=>l.type==='annotations'&&l.token===spec.layers.find(x=>x.type==='annotations')?.token);
    if(source)selectAnnotations(source,focus.features.map(f=>f.properties.review_polygon||f.properties.id),{replace:true});
    zoomTo(focus.features[0]); focus.visible=true;renderShapes();
  }
  if(spec.review_selection){
    const selection=spec.review_selection,layer=state.layers.find(l=>l.id.startsWith('review-result-')&&l.type==='cells');
    if(layer){layer.color_by='Review group';layer.filters=[{col:'Review group',values:selection.ids}];layer.hidden=new Set();layer.mode='both';setVisible(layer,true);bump(layer);}
    if(selection.bounds){const [x0,y0,x1,y1]=selection.bounds;zoomTo({geometry:{type:'Polygon',coordinates:[[[x0-10,y0-10],[x1+10,y0-10],[x1+10,y1+10],[x0-10,y1+10],[x0-10,y0-10]]]}});}
  }
  buildPanel();
  if(state.reviewer&&dataView.holder?.isConnected){const layer=state.layers.find(l=>l.type==='cells');const key=layer?.id+'|'+layer?.table;if(layer&&dataView.sourceKey!==key)await dataChoose(key);else if(!layer){dataView.source=null;dataView.grid.replaceChildren();dataDrawHead();dataView.status.textContent='No cell table was shared for this image.';}}
  return true;
}

function reviewResizeColumns(table, key) {
  const widths=state.reviewWidths||(state.reviewWidths={}), saved=widths[key]||(widths[key]=[]);
  const headings=[...table.querySelectorAll('thead th')], cols=el('colgroup');
  headings.forEach((_,i)=>cols.append(el('col',{style:'width:'+(saved[i]||(key==='image-items'?[38,135,200,130][i]:key==='shared-images'?(i===0?235:130):(i===0?38:i===1?235:150)))+'px'})));
  table.prepend(cols);
  const apply=(i,width)=>{saved[i]=Math.max(34,Math.ceil(width));cols.children[i].style.width=saved[i]+'px';};
  const fit=()=>headings.forEach((head,i)=>{
    const cells=[head,...[...table.tBodies[0]?.rows||[]].map(row=>row.cells[i])].filter(Boolean);
    const canvas=document.createElement('canvas'),context=canvas.getContext('2d');
    context.font=getComputedStyle(head).font;
    const size=Math.max(34,...cells.map(cell=>context.measureText(cell.innerText).width+32));apply(i,size);
  });
  headings.forEach((head,i)=>{
    const divider=el('span',{class:'review-column-divider',role:'separator',tabindex:0,'aria-orientation':'vertical',
      'aria-label':'Resize '+head.innerText+' column',title:'Drag to resize. Double-click to fit all columns.'});
    divider.ondblclick=event=>{event.preventDefault();event.stopPropagation();fit();};
    divider.onpointerdown=event=>{
      event.preventDefault();event.stopPropagation();divider.setPointerCapture(event.pointerId);
      const start=event.clientX,width=head.getBoundingClientRect().width;
      divider.onpointermove=e=>apply(i,width+e.clientX-start);
      divider.onpointerup=()=>{divider.onpointermove=null;divider.onpointerup=null;};
    };
    divider.onkeydown=event=>{if(event.key==='Enter'){fit();event.preventDefault();}else if(event.key==='ArrowRight'||event.key==='ArrowLeft'){apply(i,head.getBoundingClientRect().width+(event.key==='ArrowRight'?20:-20));event.preventDefault();}};
    head.append(divider);
  });
}

function reviewMetadataPicker(names,current,apply){
  const chosen=new Set(current),dialog=el('dialog',{class:'review-dialog'}),search=el('input',{type:'search',placeholder:'Find columns…','aria-label':'Find metadata columns'});
  const options=el('div',{class:'review-suggestion-options'}),all=el('input',{type:'checkbox','aria-label':'Select matching metadata columns'}),status=el('span',{class:'muted'});
  const visible=()=>[...new Set([...names,...current])].filter(n=>n.toLowerCase().includes(search.value.toLowerCase()));
  const draw=()=>{const matches=visible();all.checked=!!matches.length&&matches.every(n=>chosen.has(n));all.indeterminate=matches.some(n=>chosen.has(n))&&!all.checked;
    options.replaceChildren(...matches.map(name=>{const tick=el('input',{type:'checkbox'});tick.checked=chosen.has(name);tick.onchange=()=>{tick.checked?chosen.add(name):chosen.delete(name);draw();};return el('label',{class:'inline'},[tick,el('span',{text:name})]);}));status.textContent=chosen.size+' selected';};
  all.onchange=()=>{visible().forEach(n=>all.checked?chosen.add(n):chosen.delete(n));draw();};search.oninput=draw;
  dialog.append(el('h3',{text:'Metadata columns'}),el('div',{class:'row'},[all,search,status]),options,
    help('Choose columns from the linked image tables. One value is shown per image; Mixed indicates several values. Missing values show a dash. Up to 30 columns. Choose the shared columns separately in Share.'),
    el('div',{class:'row'},[el('button',{class:'btn primary',text:'Apply columns',onclick:async()=>{if(chosen.size>30){status.textContent='Choose up to 30 columns.';return;}await apply([...chosen]);dialog.close();}}),el('button',{class:'btn small',text:'Cancel',onclick:()=>dialog.close()})]));
  dialog.onclose=()=>dialog.remove();document.body.append(dialog);draw();dialog.showModal();
}

function reviewQueue(holder, {session, publicMode=false, openImage}={}) {
  let gone=false, selected=null, offset=0, revision=-1, serial=0, page=[], tab='items', sortKey=null, sortDirection=1;
  const checked=new Set(),checkedImages=new Set();let sharedCount=null,totalItems=0,selectionEpoch=0,refreshEpoch=0,listImageToken=null,listQueryKey=null;
  const removeButton=el('button',{class:'btn small',text:'Remove from review',disabled:'',
    title:'Remove selected entries from review and shared lists. Keep original objects, saved edits and decision history.',
    onclick:()=>removeReviewItems([...checked])});
  let previewSerial=0,previewSignature='',focusImageToken='',opening=false,pickedTimer,syncingPicks=false,initialImagePending=publicMode;
  const currentImage=()=>state.layers.find(l=>l.type==='image');
  const queueParams=()=>{
    const token=currentImage()?.token||'';
    // A legacy share may stop including an image when its last flag is removed.
    // Continue polling removals without falling back to all other images' rows.
    return new URLSearchParams({kind:kind.value,state:stateFilter.value,search:search.value,
      image:publicMode&&!imageRows.some(row=>row.roi===token)?'':token});
  };
  const destination=el('select',{'aria-label':'Send new flags to'});
  const destinationRow=el('div',{class:'row review-destination'});
  const route=publicMode ? '' : '/studio/'+encodeURIComponent(session)+'/review';
  const controls=el('div',{class:'row wrap'}), filters=el('div',{class:'row wrap'});
  const status=el('div',{class:'review-status',role:'status'}), counts=el('div',{class:'review-counts'});
  const imageList=el('div',{class:'review-images'});
  let imageRows=[], imageCounts=[], metadataColumns=[],metadataValues={},metadataEpoch=0,imageEpoch=0;
  const list=el('div',{class:'review-list'}), details=el('div',{class:'review-detail'}), pages=el('div',{class:'row'});
  const search=el('input',{type:'search',placeholder:'Find image or item…','aria-label':'Search review queue'});
  const kind=select([['','All types'],...Object.entries(REVIEW_NAMES)],'',()=>{offset=0;refresh(true);});
  const stateFilter=select([['','All states'],...Object.entries(REVIEW_STATES)],'',()=>{offset=0;refresh(true);});
  let searchTimer;search.addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{offset=0;refresh(true);},200);});
  const polygonSort=select([['place','By place'],['size_down','Largest first'],['size_up','Smallest first']],state.reviewListSort||'size_down',value=>{
    state.reviewListSort=value;sortKey=null;offset=0;refresh(true);
  });
  polygonSort.setAttribute('aria-label','Sort review list polygons');
  filters.append(search,kind,stateFilter,polygonSort);
  const setStatus=error=>status.textContent=error.message||String(error);
  function syncChecks(){
    removeButton.disabled=!checked.size;
    const all=list.querySelector('thead input[type=checkbox]');
    if(all){all.checked=!!page.length&&page.every(i=>checked.has(i.id));all.indeterminate=page.some(i=>checked.has(i.id))&&!all.checked;}
    list.querySelectorAll('tr[data-id] input[type=checkbox]').forEach(tick=>tick.checked=checked.has(tick.closest('tr').dataset.id));
    const count=pages.querySelector('span');if(count)count.textContent=totalItems+' items · '+checked.size+' selected';
  }
  async function selectMatching(){
    const sequence=++selectionEpoch,params=queueParams();
    try{const found=await getJSON(publicMode?'/selection?'+params:route+'?action=selection&'+params);
      if(gone||sequence!==selectionEpoch||params.toString()!==queueParams().toString())return;
      found.ids.forEach(id=>checked.add(id));syncChecks();
      if(!selected&&page.length)choose(page[0]);else updatePreview(true,true);
    }catch(error){setStatus(error);}
  }
  async function updatePreview(force=false,reveal=false){
    const imageToken=state.layers.find(l=>l.type==='image')?.token;
    if(!selected||focusImageToken!==imageToken)return;
    const signature=JSON.stringify([imageToken,revision]),existing=state.layers.find(l=>l.id==='review-focus');
    if(existing?.reviewPreviewKey===signature){applyPreviewSelection(existing,reveal);return;}
    if(signature===previewSignature)return;
    const request=++previewSerial,generation=state.initGeneration;previewSignature=signature;
    try{
      const found=await postJSON(publicMode?'/preview':route,{action:'preview',items:[],focus:selected.id,all_in_image:true});
      if(gone||request!==previewSerial||generation!==state.initGeneration)return;
      let layer=state.layers.find(l=>l.id==='review-focus');
      if(!layer){layer={id:'review-focus',type:'shapes',name:'Review polygons',review_inline:true,editable:false,visible:true,opacity:1,regions:true,features:[]};state.layers.push(layer);}
      layer.features=found.features;layer.reviewPreviewKey=signature;applyPreviewSelection(layer,reveal);
    }catch(error){if(request===previewSerial){previewSignature='';setStatus(error);}}
  }
  function applyPreviewSelection(layer,reveal){
    if(reveal)layer.visible=true;
    const picks=annotationSelection(layer);picks.clear();
    const sources=new Map(state.layers.filter(l=>l.token).map(l=>[l.token,l])),changed=new Set();
    for(const f of layer.features||[]){const p=f.properties,on=checked.has(p.review_item);if(on)picks.add(p.id);
      const source=sources.get(p.review_source);if(!source)continue;
      const ids=annotationSelection(source);if(on!==ids.has(p.review_polygon)){on?ids.add(p.review_polygon):ids.delete(p.review_polygon);changed.add(source);}
    }
    changed.forEach(source=>annotationOutline(source));
    syncingPicks=true;try{syncAnnotationSelection();}finally{syncingPicks=false;}buildPanel();
  }
  const imageKinds=[['annotation','Ann'],['cells','Cell labels'],['leiden','Leiden'],['som','SOM'],['vortex','Vortex'],['subcluster','Subclusters'],['neighbourhood','Neighbourhoods'],['tls','TLS'],['gland','Glands'],['niche','Niches']];
  async function chooseImage(row){
    initialImagePending=false;search.value='';offset=0;
    if(publicMode){const sequence=++serial;selected=null;details.replaceChildren();
      try{const found=await getJSON('/view_image?id='+encodeURIComponent(row.roi));if(gone||sequence!==serial)return;opening=true;try{await showReviewItem(found,()=>!gone&&sequence===serial);}finally{opening=false;}refresh(true);}catch(error){setStatus(error);}
    }else if(openImage){openImage(row.roi);refresh(true);}
  }
  function drawImages(){
    const table=el('table',{class:'studio-table review-table review-image-table'}), head=el('tr'),body=el('tbody');
    const available=imageRows.filter(r=>r.image_path);
    if(!publicMode){const all=el('input',{type:'checkbox','aria-label':'Select all images'});all.checked=!!available.length&&available.every(r=>checkedImages.has(r.roi));all.indeterminate=available.some(r=>checkedImages.has(r.roi))&&!all.checked;all.disabled=!available.length;
      all.onchange=()=>{available.forEach(r=>all.checked?checkedImages.add(r.roi):checkedImages.delete(r.roi));drawImages();};head.append(el('th',{},[all]));}
    ['Image',...metadataColumns,...imageKinds.map(([,title])=>title)].forEach(title=>head.append(el('th',{text:title})));
    for(const row of imageRows){
      const name=row.image||row.roi, tr=el('tr',{'data-review-image':row.roi,...(publicMode?{'data-view-token':row.roi}:{'data-view-path':row.image_path||''}),tabindex:0});
      if(!publicMode){const tick=el('input',{type:'checkbox','aria-label':'Share image '+name});tick.checked=checkedImages.has(row.roi);
        tick.disabled=!row.image_path;if(tick.disabled)tick.title='Assign this image in Phenotyping or Spatial analysis before sharing.';
        tick.onclick=e=>{e.stopPropagation();tick.checked?checkedImages.add(row.roi):checkedImages.delete(row.roi);drawImages();};tr.append(el('td',{},[tick]));}
      tr.append(el('td',{text:name,title:row.image_path||name}));
      for(const column of metadataColumns){const value=(publicMode?row.metadata:metadataValues[row.roi])?.[column];tr.append(el('td',{text:value?.text||'—',title:value?.detail||'No value'}));}
      for(const [type,title] of imageKinds){
        const matches=imageCounts.filter(v=>v.image===name&&(v.kind===type||v.kind===type+'_type'));
        const total=matches.reduce((n,v)=>n+v.n,0),done=matches.filter(v=>v.state==='applied').reduce((n,v)=>n+v.n,0),started=matches.some(v=>v.state!=='pending');
        tr.append(el('td',{},[el('button',{class:'btn tiny review-count '+(total?done===total?'done':started?'partial':'pending':''),
          text:total?done+'/'+total:'—',title:(REVIEW_NAMES[type]||title)+': '+done+' applied / '+total+' flagged',
          onclick:event=>{event.stopPropagation();kind.value=type;chooseImage(row);}})]));
      }
      tr.onclick=()=>chooseImage(row);tr.onkeydown=event=>{if(event.key==='Enter')chooseImage(row);};body.append(tr);
    }
    table.append(el('thead',{},[head]),body);const left=imageList.scrollLeft,top=imageList.scrollTop;imageList.replaceChildren(table);reviewResizeColumns(table,publicMode?'shared-images':'images');imageList.scrollLeft=left;imageList.scrollTop=top;
  }
  async function loadImages(){
    const sequence=++imageEpoch,metadataSequence=metadataEpoch;
    try{const found=await getJSON(publicMode?'/images':route+'?action=images');if(gone||sequence!==imageEpoch)return;imageRows=found.images;imageCounts=found.counts;
      if(publicMode||metadataSequence===metadataEpoch)metadataColumns=found.metadata_columns||[];
      drawImages();if(!publicMode&&metadataColumns.length&&metadataSequence===metadataEpoch)loadMetadata(metadataColumns);
      if(initialImagePending&&!currentImage()&&imageRows.length){initialImagePending=false;chooseImage(imageRows.find(row=>imageCounts.some(c=>c.image===row.image&&c.n))||imageRows[0]);}}
    catch(error){setStatus(error);}
  }
  async function loadMetadata(columns,save=false){
    const sequence=++metadataEpoch;
    try{const found=await postJSON(route,{action:'metadata',columns,save});if(gone||sequence!==metadataEpoch)return;
      metadataColumns=found.columns;metadataValues=found.values;drawImages();
    }catch(error){setStatus(error);}
  }
  async function pickMetadata(current,apply){
    try{const found=await getJSON(route+'?action=metadata_options');reviewMetadataPicker(found.columns,current,apply);}catch(error){setStatus(error);}
  }
  async function action(body) {
    try { const result=await postJSON(publicMode?'/decision':route,body); if(result.id===selected?.id)selected=result;
      await refresh(true); if(selected)renderDecision(selected); return result;
    } catch(error){setStatus(error);}
  }
  async function rowDecisionTargets(item){
    if(!checked.has(item.id))return [item];
    const token=currentImage()?.token||'';
    const params=new URLSearchParams({image:token,kind:item.kind});
    const matching=await getJSON(publicMode?'/selection?'+params:route+'?action=selection&'+params);
    if(token!==currentImage()?.token)throw Error('The image changed. Select the polygons again.');
    const ids=matching.ids.filter(id=>checked.has(id));
    if(!ids.length)return [];
    const found=await postJSON(publicMode?'/label_options':route,{action:'suggestion_options',ids});
    if(gone||token!==(currentImage()?.token||''))throw Error('The image changed. Select the polygons again.');
    return found.items.filter(i=>!['applied','applying','requested'].includes(i.state));
  }
  async function choose(item) {
    checked.add(item.id);syncChecks();
    selected=item; renderDecision(item);
    const sequence=++serial;
    try {
      const found=await getJSON(publicMode?'/view?id='+item.id:route+'?action=view&id='+item.id);
      if(gone||sequence!==serial)return;
      opening=true;
      try{if(!await showReviewItem(found,()=>!gone&&sequence===serial))return;focusImageToken=found.spec.layers.find(l=>l.type==='image')?.token;previewSignature='';}finally{opening=false;}
      await updatePreview(true,true);
      item.contextToken=found.spec.layers.find(layer=>layer.type==='annotations')?.token;
      syncPicked();
      list.querySelectorAll('tr[data-id]').forEach(row=>row.classList.toggle('on',row.dataset.id===item.id));
    }catch(error){setStatus(error);}
  }
  function renderDecision(item) {
    details.replaceChildren();
    if(!item)return;
    const label=el('input',{value:item.decision?.label||'',placeholder:'Another label…','aria-label':'Review label'});
    const note=el('textarea',{rows:2,value:item.decision?.note||'',placeholder:'Optional comment','aria-label':'Decision comment'});
    note.value=item.decision?.note||'';
    const decide=async decision=>{
      const targets=item.kind==='annotation'?reviewDecisionTargets(item):await rowDecisionTargets(item);
      if(targets.length>1)return decideReviewItems(targets,decision,label.value,note.value);
      return action({action:'decide',id:item.id,version:item.version,decision,label:label.value,note:note.value});
    };
    const choices=el('div',{class:'row wrap'});
    (item.suggestions||[]).forEach(name=>choices.append(el('button',{class:'btn small',text:name,onclick:()=>{label.value=name;decide('relabel');}})));
    choices.append(label,el('button',{class:'btn small',text:'Use label',onclick:()=>decide('relabel')}));
    const actions=el('div',{class:'row wrap'},['keep','reject','unable'].map((v,i)=>el('button',{class:'btn small',text:['Keep','Reject','Unable to decide'][i],onclick:()=>decide(v)})));
    if(item.state==='applied'||item.state==='applying') { choices.querySelectorAll('button,input').forEach(b=>b.disabled=true); actions.querySelectorAll('button').forEach(b=>b.disabled=true); }
    if(!publicMode){
      actions.append(el('button',{class:'btn small',text:'Suggestions…',onclick:()=>reviewSuggestionDialog([item.id])}));
      if(item.state==='requested')actions.append(el('button',{class:'btn small',text:'Approve nomination',onclick:()=>action({action:'reopen',id:item.id,version:item.version})}));
      actions.append(el('button',{class:'btn primary small',text:'Accept & apply',disabled:item.state==='answered'?null:'',
        onclick:()=>action({action:'apply',id:item.id,version:item.version})}),
        el('button',{class:'btn small',text:'Reopen',onclick:()=>action({action:'reopen',id:item.id,version:item.version})}));
    }
    const next=()=>{const index=page.findIndex(i=>i.id===item.id); if(page[index+1])choose(page[index+1]);};
    actions.append(el('button',{class:'btn small',text:'Next →',onclick:next}));
    details.append(el('div',{class:'row'},[el('strong',{text:item.title}),el('span',{class:'muted',text:REVIEW_STATES[item.state]})]));
    if(item.note)details.append(el('div',{class:'muted',text:item.note}));
    if(item.decision)details.append(el('div',{text:'Decision: '+(item.decision.label||item.decision.action)+(item.decision.note?' · '+item.decision.note:'')}));
    if(item.applied?.error||item.applied?.message)details.append(el('div',{role:'status',text:item.applied.error||item.applied.message}));
    details.append(choices,note,actions);
    if(item.has_result_context || item.target?.job_id)renderResultContext(item);
    if(publicMode && item.kind==='annotation'){
      const context=el('details'), list=el('div');let contextOffset=0;
      context.append(el('summary',{text:'All polygons in this source'}),list);details.append(context);
      async function loadContext(){try{
        const found=await getJSON('/components?item='+item.id+'&offset='+contextOffset);list.replaceChildren();
        found.rows.forEach(row=>list.append(el('div',{class:'row'},[el('span',{class:'grow',text:row.name}),
          el('button',{class:'btn tiny',text:'Go',onclick:()=>{const [x0,y0,x1,y1]=row.bounds;zoomTo({geometry:{type:'Polygon',coordinates:[[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]}});}}),
          el('button',{class:'btn tiny',text:'Suggest review',onclick:async e=>{try{await postJSON('/nominate',{item:item.id,ids:[row.id],note:note.value});e.target.textContent='Sent to owner';e.target.disabled=true;}catch(error){setStatus(error);}}})])));
        list.append(el('div',{class:'row'},[el('button',{class:'btn tiny',text:'←',disabled:contextOffset?null:'',onclick:()=>{contextOffset=Math.max(0,contextOffset-100);loadContext();}}),
          el('span',{text:found.total+' polygons'}),el('button',{class:'btn tiny',text:'→',disabled:contextOffset+found.rows.length<found.total?null:'',onclick:()=>{contextOffset+=100;loadContext();}})]));
      }catch(error){setStatus(error);}}
      context.ontoggle=()=>{if(context.open)loadContext();};
    }
  }
  async function renderResultContext(item){
    const box=el('details',{open:true,class:'review-result-context'},[el('summary',{text:'Full result · groups and labels'})]);
    const host=el('div',{class:'studio-pairs-scroll'});box.append(host);details.append(box);
    try{
      const data=await getJSON(publicMode?'/result_context?id='+item.id:route+'?action=result_context&id='+item.id);
      if(!box.isConnected||selected?.id!==item.id)return;
      const profile=data.profile||{},values=profile.z||profile.values||[],columns=profile.columns||[],names=profile.rows||[];
      const scale=Math.max(.1,...values.flat().filter(Number.isFinite).map(Math.abs));
      const profileCells=(row,index)=>columns.forEach((name,j)=>{
        const z=values[index]?.[j],raw=profile.raw?.[index]?.[j];
        row.append(el('td',{
          text:Number.isFinite(raw)?(profile.z?(100*raw).toFixed(1)+'%':Number(raw.toPrecision(3)).toString()):Number.isFinite(z)?z.toFixed(2):'—',
          title:name+(Number.isFinite(z)?' · Z '+z.toFixed(2):''),
          style:Number.isFinite(z)?'background:rgba('+(z>0?'190,55,65':'45,110,165')+','+(Math.abs(z)/scale*.85)+');text-align:center':''}));
      });
      if(data.individual&&columns.length){
        const profiles=el('table',{class:'studio-counts review-result-heatmap'}),rows=el('tbody');
        profiles.append(el('thead',{},[el('tr',{},['Type',...columns].map(text=>el('th',{text})))]),rows);
        names.forEach((name,index)=>{const row=el('tr',{},[el('td',{text:name})]);profileCells(row,index);rows.append(row);});
        host.append(profiles);
      }
      const table=el('table',{class:'studio-counts review-result-heatmap'}),body=el('tbody'),pager=el('div',{class:'row'});
      table.append(el('thead',{},[el('tr',{},[data.individual?'Structure':'Group','Saved label','Review','Cells',...(data.individual?[]:columns),''].map(text=>el('th',{text})))]),body);
      let offset=0;
      function drawGroups(){
      body.replaceChildren();pager.replaceChildren();
      for(const group of data.groups.slice(offset,offset+100)){
        const index=names.indexOf(group.name),on=data.selected.includes(group.name);
        const row=el('tr',{class:on?'selected':'','data-review-group':group.name});
        const show=()=>{const layer=state.layers.find(l=>l.id.startsWith('review-result-')&&l.type==='cells');
          if(!layer){message('Open the review item to show its cells.');return;}
          layer.color_by='Review group';layer.filters=[{col:'Review group',values:[group.name]}];layer.hidden=new Set();
          layer.mode='both';setVisible(layer,true);bump(layer);buildPanel();};
        row.append(el('td',{},[el('button',{class:'studio-item-name',text:group.title,disabled:group.cells?null:'',onclick:show})]),
          el('td',{text:group.label||'Unnamed'}),el('td',{text:REVIEW_STATES[group.state]||group.state}),el('td',{text:group.cells.toLocaleString()}));
        if(!data.individual)profileCells(row,index);
        row.append(el('td',{},publicMode&&group.cells&&group.state==='Not flagged'?[el('button',{class:'btn tiny',text:'Suggest review',onclick:async e=>{
          try{await postJSON('/nominate_result',{id:item.id,groups:[group.name]});e.target.textContent='Sent to owner';e.target.disabled=true;}catch(error){setStatus(error);}}})]:[]));
        body.append(row);
      }
      if(data.groups.length>100)pager.append(
        el('button',{class:'btn tiny',text:'←',disabled:offset?null:'',onclick:()=>{offset-=100;drawGroups();}}),
        el('span',{text:(offset+1)+'–'+Math.min(offset+100,data.groups.length)+' / '+data.groups.length}),
        el('button',{class:'btn tiny',text:'→',disabled:offset+100<data.groups.length?null:'',onclick:()=>{offset+=100;drawGroups();}}));
      }
      drawGroups();host.append(table);box.append(pager);if(profile.note)box.append(help(profile.note));
    }catch(error){host.textContent=error.message;}
  }
  function syncPicked(){
    const groups=annotationSelectedGroups();
    const ids=new Set(groups.flatMap(([,members])=>[...members]));
    document.querySelectorAll('[data-review-feature] input[type=checkbox]').forEach(tick=>{
      const row=tick.closest('[data-review-feature]');tick.checked=ids.has(row.dataset.reviewFeature);
    });
    list.querySelectorAll('tr[data-id]').forEach(row=>{
      const item=page.find(i=>i.id===row.dataset.id);
      const own=item?.target?.ids||item?.polygon_ids||[];
      const on=groups.some(([layer,members])=>layer.review_inline?(layer.features||[]).some(f=>f.properties.review_item===item.id&&members.has(f.properties.id)):own.some(id=>members.has(id))&&
        (!item?.target?.source||layer.token===item.contextToken||layer.saved_file===item.target.source||layer.review_item===item.id));
      row.classList.toggle('polygon-picked',on);
    });
  }
  const pickedChanged=()=>{
    syncPicked();if(opening||syncingPicks)return;
    clearTimeout(pickedTimer);pickedTimer=setTimeout(()=>{
      if(gone||opening)return;
      const groups=annotationSelectedGroups();
      const current=new Set(),chosen=new Set(),sources=new Map(groups.filter(([l])=>l.token).map(([l,ids])=>[l.token,ids]));
      for(const layer of state.layers.filter(l=>l.review_inline))for(const f of layer.features||[]){const p=f.properties;current.add(p.review_item);
        if(annotationSelection(layer).has(p.id)||sources.get(p.review_source)?.has(p.review_polygon))chosen.add(p.review_item);
      }
      current.forEach(id=>chosen.has(id)?checked.add(id):checked.delete(id));
      ++selectionEpoch;syncChecks();
      if(selected)updatePreview(true,true);
    },0);
  };
  document.addEventListener('spacemap-annotations-picked',pickedChanged);
  const contextDecision=event=>{
    if(!publicMode||gone)return;
    selected=event.detail;renderDecision(selected);
    list.querySelectorAll('tr[data-id]').forEach(row=>row.classList.toggle('on',row.dataset.id===selected.id));
    refresh(true);
    updatePreview(true);
  };
  document.addEventListener('spacemap-review-context',contextDecision);
  async function deleteLink(id){
    await postJSON(route,{action:'delete_link',id});
    await loadDestination();if(tab==='links')await links();
  }
  async function share(initial='') {
    const selectedItems=[...checked],selectedImages=[...checkedImages],reviewKind=kind.value;
    const modal=el('dialog',{class:'review-dialog'}), name=el('input',{placeholder:'Reviewer or specialty','aria-label':'Review link name'});
    const saved=el('select',{'aria-label':'Saved reviewer link'}), output=el('div',{role:'status'});
    const access=select([['internal','Internal / VPN'],['external','HTTPS · external browser']], 'internal',()=>{});
    access.setAttribute('aria-label','Review link access');
    const days=el('input',{type:'number',min:1,max:90,value:14,'aria-label':'Link expiry in days'});
    const expiry=el('span',{class:'muted'}), dayRow=el('label',{class:'row'},[days,el('span',{text:'days'})]);
    const legacy=el('input',{placeholder:'Paste the original link once…','aria-label':'Original reviewer link'});legacy.hidden=true;
    const include=el('input',{type:'checkbox',checked:selectedItems.length||selectedImages.length?'':null,'aria-label':'Add selection to this link'});
    const includeText=el('span'),includeRow=el('label',{class:'row'},[include,includeText]);
    const scope=select([
      ...(selectedImages.length?[['images',(REVIEW_NAMES[reviewKind]||'All review types')+' in selected images']]:[]),
      ['items','Selected review items ('+selectedItems.length.toLocaleString()+')'],
      ...(selectedImages.length?[['none','Images only · no review items']]:[])
    ],selectedImages.length?'images':'items',()=>syncScope());
    scope.setAttribute('aria-label','Review items to share');
    const scopeRow=el('div',{class:'row'},[el('label',{text:'Review items'}),scope]);
    let imageItems=null,selectionError='',saving=false,savedLinks=[],shareColumns=[],linkChoice=0;
    const chosenItems=()=>scope.value==='images'?imageItems||[]:scope.value==='items'?selectedItems:[];
    function syncScope(){
      const link=savedLinks.find(l=>l.id===saved.value),adding=!link||include.checked;
      const pending=scope.value==='images'&&imageItems===null;
      includeText.textContent=(pending?'Counting review items…': 'Add '+chosenItems().length.toLocaleString()+' review items')+' · '+selectedImages.length+' images';
      scope.disabled=!!link&&!include.checked||saving;
      make.disabled=saving||!!link&&(link.revoked||link.expires*1000<=Date.now())||
        adding&&pending||!link&&!selectedImages.length&&!chosenItems().length;
      if(adding&&pending&&selectionError)output.textContent=selectionError;
    }
    include.onchange=syncScope;
    function displayURL(url,summary=''){
      legacy.hidden=false;legacy.readOnly=true;legacy.value=url;legacy.setAttribute('aria-label','Review link');
      output.replaceChildren(...(summary?[el('div',{text:summary})]:[]),el('a',{href:url,target:'_blank',rel:'noopener noreferrer',text:'Open reviewer page'}));
    }
    const metadata=el('button',{class:'btn small',text:'Shared metadata: none',onclick:()=>pickMetadata(shareColumns,columns=>{shareColumns=columns;metadata.textContent='Shared metadata: '+(columns.length||'none');})});
    const make=el('button',{class:'btn primary',text:'Create link',disabled:'',onclick:async()=>{
      saving=true;syncScope();output.replaceChildren();
      const request=++linkChoice;
      try {
        const existing=saved.value;
        const found=await postJSON(route,{action:existing?'update_link':'share',id:existing,name:name.value,access:access.value,
          ...(existing?{}:{days:Number(days.value)}),existing_url:legacy.value,
          items:!existing||include.checked?chosenItems():[],images:!existing||include.checked?selectedImages:[],metadata_columns:shareColumns,cell_context:reviewCellContext()});
        if(!modal.isConnected||request!==linkChoice)return;
        await reload(found.id);loadDestination();if(tab==='links')links();
        displayURL(found.url,name.value+' · '+Number(found.total).toLocaleString()+' review items shared');
      }catch(error){output.textContent=error.message;}finally{saving=false;syncScope();}
    }});
    const removeLink=el('button',{class:'btn small',text:'Delete link',hidden:'',
      title:'Disable this link and remove it from the list. Review items and decisions are kept.',onclick:async()=>{
        const id=saved.value;if(!id||saving)return;
        saving=true;syncScope();removeLink.disabled=true;++linkChoice;
        try{await deleteLink(id);await reload('');output.textContent='Link deleted. Review items and decisions kept.';}
        catch(error){output.textContent=error.message;}
        finally{saving=false;removeLink.disabled=false;syncScope();}
      }});
    async function chooseLink(preserveNew=false){
      const request=++linkChoice;
      const link=savedLinks.find(l=>l.id===saved.value),closed=link&&(link.revoked||link.expires*1000<=Date.now());
      removeLink.hidden=!link;
      if(link||!preserveNew){name.value=link?.name||'';access.value=link?.access||'internal';}
      legacy.hidden=!link||closed;legacy.readOnly=!!link?.url_saved;legacy.value='';
      legacy.placeholder=link?.url_saved?'Loading saved link…':'Paste the original link once…';legacy.setAttribute('aria-label','Original reviewer link');
      dayRow.hidden=!!link;expiry.textContent=link?'Expires '+new Date(link.expires*1000).toLocaleDateString():'';
      if(link||!preserveNew)shareColumns=[...(link?.scope?.context?.metadata_columns||[])];metadata.textContent='Shared metadata: '+(shareColumns.length||'none');
      includeRow.hidden=!link||!(selectedItems.length||selectedImages.length);make.textContent=link?'Update link':'Create link';syncScope();
      output.textContent=closed?'This link is closed. Choose New link to share again.':'';
      if(link&&!closed){
        try{const found=await getJSON(route+'?action=link&id='+encodeURIComponent(link.id));
          if(modal.isConnected&&request===linkChoice)displayURL(found.url);
        }catch(error){if(modal.isConnected&&request===linkChoice){legacy.readOnly=false;legacy.placeholder='Paste the original link once…';output.textContent=error.message;}}
      }
    }
    async function reload(id){
      const found=await getJSON(route+'?action=links');savedLinks=found.links;
      saved.replaceChildren(el('option',{value:'',text:'New link…'}),...savedLinks.map(l=>el('option',{value:l.id,text:l.name+(l.revoked||l.expires*1000<=Date.now()?' · Closed':'')})));
      saved.value=id||'';await chooseLink(!id);
    }
    saved.onchange=()=>chooseLink();
    modal.append(el('h3',{text:'Share images and review items'}),saved,name,legacy,el('div',{class:'row'},[access,dayRow,expiry]),scopeRow,includeRow,metadata,
      help('With images checked, include their flagged review items across every page, limited to the current review type. Selected review items uses only the checked items in the lower list. Images only shares the images without adding flags. Updating a link adds to its existing list and preserves decisions.'),
      help('Only metadata chosen here is included in the image list. Image metadata is a snapshot when the link is saved; Mixed means that image has several values. This choice does not redact columns from a separately shared cell table. Images may be shared without review items.'),
      help('Saved links appear automatically. Update link refreshes the address for the current node or configured HTTPS service while keeping the same review list and decisions. Send the updated address if the old node is unavailable. An older link with no saved credential can recover itself when the reviewer opens it, or you can paste it once. Anyone holding an active link can submit decisions; source changes still require owner approval.'),
      el('div',{class:'row'},[make,removeLink,el('button',{class:'btn',text:'Close',onclick:()=>{modal.close();modal.remove();}})]),output);
    document.body.append(modal);modal.showModal();
    if(selectedImages.length){
      postJSON(route,{action:'share_selection',images:selectedImages,kind:reviewKind}).then(found=>{
        if(!modal.isConnected)return;imageItems=found.ids;syncScope();
      }).catch(error=>{if(modal.isConnected){selectionError=error.message;syncScope();}});
    }
    try{await reload(initial);}catch(error){output.textContent=error.message;}
  }
  async function loadDestination(){
    if(publicMode)return;
    try{const found=await getJSON(route+'?action=links');if(gone)return;
      destination.replaceChildren(el('option',{value:'',text:'No reviewer yet'}),...found.links.filter(l=>!l.revoked&&l.expires*1000>Date.now()).map(l=>el('option',{value:l.id,text:l.name})));
      destination.value=found.destination||'';
    }catch(error){setStatus(error);}
  }
  destination.onchange=async()=>{try{await postJSON(route,{action:'destination',id:destination.value});}catch(error){setStatus(error);loadDestination();}};
  if(!publicMode)destinationRow.append(el('label',{text:'Send new flags to'}),destination,
    el('button',{class:'btn small',text:'Add selected',onclick:async()=>{try{if(!destination.value)throw Error('Choose a reviewer link first.');if(!checked.size)throw Error('Select flagged items to add.');const found=await postJSON(route,{action:'add_to_link',id:destination.value,items:[...checked]});status.textContent=found.added+' items added to the same link.';if(tab==='links')links();}catch(error){setStatus(error);}}}),
    help('New owner flags automatically join this named reviewer’s existing link. Add selected also sends already-flagged items. Other links stay unchanged; expired or revoked links stop receiving items.'));
  async function links() {
    try {const found=await getJSON(route+'?action=links');list.replaceChildren();
      found.links.forEach(link=>{
        const expired=link.revoked||link.expires*1000<Date.now();
        list.append(el('details',{class:'review-link'},[el('summary',{text:link.name+' · '+link.scope.items.length+' items · '+Object.keys(link.scope.context?.images||{}).length+' images · '+(expired?'Closed':'Active')}),
          el('div',{class:'row'},[el('span',{text:'Expires '+new Date(link.expires*1000).toLocaleDateString()}),
            el('button',{class:'btn small',text:'Edit link…',onclick:()=>share(link.id)}),
            el('button',{class:'btn small',text:'Revoke',disabled:expired?'':null,onclick:async()=>{await postJSON(route,{action:'revoke',id:link.id});links();loadDestination();}}),
            el('button',{class:'btn small',text:'Delete link',title:'Disable this link and remove it from the list. Review items and decisions are kept.',onclick:async event=>{
              event.currentTarget.disabled=true;try{await deleteLink(link.id);}catch(error){setStatus(error);links();}
            }})])]));
      });
    }catch(error){setStatus(error);}
  }
  async function refresh(force=false) {
    if(gone)return;if(tab==='links')return;
    const sequence=++refreshEpoch;
    try {
      const params=queueParams();params.set('offset',String(offset));params.set('sort',polygonSort.value);
      const queryKey=params.toString();params.set('since',String(revision));
      const found=await getJSON(publicMode?'/queue?'+params:route+'?'+params);
      if(gone||sequence!==refreshEpoch||params.get('image')!==queueParams().get('image'))return;
      forgetItems(found.removed);
      if(publicMode){
        if(sharedCount!==null && found.shared_count>sharedCount)message((found.shared_count-sharedCount)+' new review items added.');
        sharedCount=found.shared_count;
      }
      if(!force&&found.revision===revision&&queryKey===listQueryKey)return;
      const changed=found.revision!==revision;revision=found.revision; page=found.items;totalItems=found.total;listImageToken=params.get('image');listQueryKey=queryKey;
      if(offset&&offset>=totalItems){offset=Math.max(0,Math.floor((totalItems-1)/100)*100);refresh(true);return;}
      if(changed&&publicMode)loadImages();
      counts.replaceChildren();
      const categories=new Map();found.counts.forEach(r=>{const c=categories.get(r.kind)||{total:0,applied:0,started:0};c.total+=r.n;if(r.state==='applied')c.applied+=r.n;if(r.state!=='pending')c.started+=r.n;categories.set(r.kind,c);});
      categories.forEach((c,k)=>counts.append(el('button',{class:'btn small review-count '+(c.applied===c.total?'done':c.started?'partial':'pending'),
        text:(REVIEW_NAMES[k]||k)+' '+c.applied+'/'+c.total,title:'Applied / total',onclick:()=>{kind.value=k;offset=0;refresh(true);}})));
      const table=el('table',{class:'studio-table review-table'}),head=el('tr');
      const all=el('input',{type:'checkbox','aria-label':'Select this page',onchange:()=>{++selectionEpoch;page.forEach(i=>all.checked?checked.add(i.id):checked.delete(i.id));syncChecks();if(all.checked&&!selected&&page.length)choose(page[0]);else updatePreview(true,true);}});
      all.checked=page.length>0&&page.every(i=>checked.has(i.id));all.indeterminate=page.some(i=>checked.has(i.id))&&!all.checked;
      head.append(el('th',{},[all]));
      const heading=(key,title)=>el('th',{},[el('button',{class:'btn small ghost',text:title,onclick:()=>{sortDirection=sortKey===key?-sortDirection:1;sortKey=key;drawRows();}})]);
      head.append(heading('title','Item'),el('th',{text:'Label'}),heading('state','State'));
      const tbody=el('tbody');table.append(el('thead',{},[head]),tbody);
      function drawRows(){tbody.replaceChildren();if(sortKey)page.sort((a,b)=>sortDirection*String(a[sortKey]).localeCompare(String(b[sortKey]),undefined,{numeric:true}));page.forEach(item=>{
        const tick=el('input',{type:'checkbox','aria-label':'Select '+item.title});tick.checked=checked.has(item.id);
        tick.onclick=e=>{e.stopPropagation();++selectionEpoch;tick.checked?checked.add(item.id):checked.delete(item.id);syncChecks();if(tick.checked&&!selected)choose(item);else updatePreview(true,true);};
        const row=el('tr',{'data-id':item.id,tabindex:0,class:item.id===selected?.id?'on':'',onclick:()=>choose(item)},[
          el('td',{},[tick]),el('td',{text:item.title,title:item.title}),
          el('td',{class:'review-row-choices'},[reviewItemChoices(item,()=>rowDecisionTargets(item))]),
          el('td',{text:REVIEW_STATES[item.state]})]);
        row.onkeydown=e=>{if(e.key==='Enter'&&e.target===row)choose(item);};tbody.append(row);
      });}
      drawRows();list.replaceChildren(table);reviewResizeColumns(table,'image-items');syncPicked();pages.replaceChildren(el('span',{class:'muted',text:found.total+' items · '+checked.size+' selected'}),
        el('button',{class:'btn small',text:'Select all matching',onclick:selectMatching}),
        el('button',{class:'btn small',text:'Clear selection',onclick:()=>{++selectionEpoch;checked.clear();syncChecks();updatePreview(true,true);}}),
        el('button',{class:'btn small',text:'←',disabled:offset?null:'',onclick:()=>{offset=Math.max(0,offset-100);refresh(true);}}),
        el('button',{class:'btn small',text:'→',disabled:offset+page.length<found.total?null:'',onclick:()=>{offset+=100;refresh(true);}}));
      const updated=selected&&page.find(i=>i.id===selected.id);
      if(updated&&updated.version!==selected.version){selected=updated;renderDecision(updated);}
      if(changed)updatePreview();
      status.textContent=found.total?'':currentImage()?'No review items for this image.':'Choose an image above to see its review items.';
    }catch(error){setStatus(error);if(publicMode)loadImages();}
  }
  if(!publicMode)controls.append(el('button',{class:'btn small',text:'Items',onclick:()=>{tab='items';filters.hidden=false;refresh(true);}}),
    el('button',{class:'btn small',text:'Links',onclick:()=>{tab='links';filters.hidden=true;details.replaceChildren();links();}}),
    el('button',{class:'btn small',text:'Flag selected',onclick:()=>flagSelectedAnnotations()}),
    el('button',{class:'btn small',text:'Suggestions…',onclick:()=>reviewSuggestionDialog(checked.size?[...checked]:checkedImages.size?[]:selected?[selected.id]:state.lastFlaggedReview||[],{images:imageRows.filter(row=>checkedImages.has(row.roi)&&row.image_path).map(row=>row.image_path),kind:kind.value||'annotation'})}),
    el('button',{class:'btn small',text:'Share…',onclick:()=>share()}),removeButton);
  if(!publicMode)controls.append(el('button',{class:'btn small',text:'Metadata columns…',onclick:()=>pickMetadata(metadataColumns,columns=>loadMetadata(columns,true))}));
  controls.append(help('Select an item to see its image. Keep, suggest a label, or reject it. Review decisions are separate from project edits; Accept & apply saves them. Red: waiting. Orange: started. Green: all applied.'));
  holder.replaceChildren(controls,...(publicMode?[]:[destinationRow]),filters,imageList,counts,list,pages,details,status);refresh(true);loadImages();loadDestination();
  function forgetItems(ids){
    if(!ids?.length)return;
    ++selectionEpoch;++previewSerial;previewSignature='';
    ids.forEach(id=>checked.delete(id));
    if(selected&&ids.includes(selected.id)){++serial;selected=null;details.replaceChildren();}
    forgetReviewPreviews(ids);syncChecks();
  }
  const stopImageHighlight=watchOpenImageRows(imageList);
  const imageOpened=()=>{if(gone)return;offset=0;sortKey=null;search.value='';if(!opening){selected=null;details.replaceChildren();}
    if(tab==='items'&&listImageToken!==(currentImage()?.token||'')){
      // Old-image rows must not remain actionable while the new queue loads.
      page=[];list.replaceChildren();counts.replaceChildren();pages.replaceChildren();
    }
    ++refreshEpoch;refresh(true);
  };
  document.addEventListener('spacemap-image-opened',imageOpened);
  const timer=setInterval(()=>refresh(),3000), changed=event=>{forgetItems(event.detail?.removed);refresh(true);loadImages();loadDestination();};
  document.addEventListener('spacemap-review-changed',changed);
  return {dispose:()=>{gone=true;stopImageHighlight();clearInterval(timer);clearTimeout(searchTimer);clearTimeout(pickedTimer);document.removeEventListener('spacemap-image-opened',imageOpened);document.removeEventListener('spacemap-review-changed',changed);document.removeEventListener('spacemap-annotations-picked',pickedChanged);document.removeEventListener('spacemap-review-context',contextDecision);}};
}

function reviewFeatureItem(feature) {
  const p=feature.properties;
  return {id:p.review_item,version:p.review_version,state:p.review_state,suggestions:p.review_suggestions||[],
    decision:p.review_decision,title:p.classification?.name||p.name};
}

function selectedReviewItems(hit) {
  const items=new Map(), selected=annotationSelection(hit.layer).has(String(hit.feature.properties.id));
  for(const layer of state.layers.filter(l=>l.review_inline))for(const feature of layer.features||[]){
    const p=feature.properties, same=layer===hit.layer&&p.id===hit.feature.properties.id ||
      p.review_source===hit.layer.token&&p.review_polygon===String(hit.feature.properties.id);
    if(same || selected&&annotationSelection(layer).has(p.id))items.set(p.review_item,reviewFeatureItem(feature));
  }
  return [...items.values()];
}

async function decideReviewItems(items, decision, label='', note) {
  if(!items.length){message('No editable review polygons selected.',true);return false;}
  state.reviewDecisionsPending ||= new Set();
  if(items.some(item=>state.reviewDecisionsPending.has(item.id)))return false;
  items.forEach(item=>state.reviewDecisionsPending.add(item.id));
  try {
    const result=await postJSON(state.reviewer?'/decisions':reviewRoute(),{action:'decide_many',
      items:Object.fromEntries(items.map(item=>[item.id,item.version])),decision,label,...(note===undefined?{}:{note})});
    document.dispatchEvent(new CustomEvent('spacemap-review-changed'));
    message(result.items.length+' decision'+(result.items.length===1?'':'s')+(state.reviewer?' sent to owner.':' recorded. Accept & apply in Review to save to the annotations.'));
    return true;
  }catch(error){message(error.message,true);document.dispatchEvent(new CustomEvent('spacemap-review-changed'));return false;}
  finally{items.forEach(item=>state.reviewDecisionsPending.delete(item.id));}
}

function reviewLabelDialog(items) {
  const dialog=el('dialog',{class:'review-dialog'}),status=el('div',{role:'status'});
  const label=el('input',{'aria-label':'Review label',placeholder:'Choose or enter a label',maxlength:500}),choices=el('div',{class:'review-label-options'});
  const note=el('textarea',{'aria-label':'Review comment',rows:2,placeholder:'Optional comment'});let noteChanged=false;
  if(items.length===1)note.value=items[0].decision?.note||'';
  note.oninput=()=>{noteChanged=true;};
  let names=[...new Set(items.flatMap(item=>item.suggestions||[]))];
  const draw=()=>choices.replaceChildren(...names.filter(name=>name.toLowerCase().includes(label.value.toLowerCase())).map(name=>
    el('button',{class:'btn small',text:name,onclick:()=>{label.value=name;draw();}})));
  label.oninput=draw;
  const save=el('button',{class:'btn primary',text:'Use label',onclick:async()=>{
    if(!label.value.trim())return label.focus();save.disabled=true;
    if(await decideReviewItems(items,'relabel',label.value.trim(),noteChanged?note.value:undefined))dialog.close();
    else{status.textContent='No decisions were changed. Refresh the selection and try again.';save.disabled=false;}
  }});
  label.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();save.click();}};
  dialog.append(el('h3',{text:'Label '+items.length+' review item'+(items.length===1?'':'s')}),label,choices,note,status,
    el('div',{class:'row'},[save,el('button',{class:'btn',text:'Cancel',onclick:()=>dialog.close()})]));
  dialog.addEventListener('close',()=>dialog.remove());document.body.append(dialog);dialog.showModal();draw();label.focus();
  postJSON(state.reviewer?'/label_options':reviewRoute(),{action:'suggestion_options',ids:items.map(item=>item.id)}).then(found=>{
    if(dialog.isConnected){names=[...new Set([...names,...found.options])];draw();}
  }).catch(error=>{if(dialog.isConnected)status.textContent=error.message;});
}

function reviewDecisionTargets(item) {
  // Resolve the selection when clicked, not when the row was built. Checkbox and
  // viewer selections change without rebuilding these compact row controls.
  const layer=state.layers.find(l=>l.review_inline&&l.features?.some(f=>f.properties.review_item===item.id));
  const current=layer?.features.find(f=>f.properties.review_item===item.id);
  return (current?selectedReviewItems({layer,feature:current}):[item])
    .filter(i=>!['applied','applying','requested'].includes(i.state));
}

function reviewPolygonChoices(feature) {
  const item=reviewFeatureItem(feature);
  return reviewItemChoices(item,()=>reviewDecisionTargets(item));
}

function reviewItemChoices(item,targets) {
  const locked=['applied','applying','requested'].includes(item.state),choices=el('span',{class:'review-polygon-choices'});
  choices.onclick=event=>event.stopPropagation();
  const run=async(action)=>{try{await action(await targets());}catch(error){message(error.message,true);}};
  for(const name of item.suggestions||[])choices.append(el('button',{class:'btn tiny',text:name,
    'aria-label':'Use '+name+' for this review item','aria-pressed':String(item.decision?.action==='relabel'&&item.decision.label===name),
    title:'Apply to this item, or all selected items of this type when this polygon is selected. Completed items stay unchanged.',
    disabled:locked?'':null,onclick:()=>run(items=>decideReviewItems(items,'relabel',name))}));
  choices.append(el('button',{class:'btn tiny',text:'+',title:'Choose another label for this item, or all selected items of this type when it is selected',
    'aria-label':'Another review label',disabled:locked?'':null,onclick:()=>run(reviewLabelDialog)}));
  return choices;
}

async function reviewPolygonMenu(actions,hit,menu,act){
  const generation=state.initGeneration, layer=hit.layer;
  let items=selectedReviewItems(hit);
  if(!items.length && state.reviewer){
    const itemId=hit.feature.properties.review_item||layer.review_item,polygonId=hit.feature.properties.review_polygon||hit.feature.properties.id;
    if(!itemId)return;
    actions.append(el('div',{class:'ctx-title',text:'Review · loading…'}));
    const found=await getJSON('/polygon?'+new URLSearchParams({item:itemId,id:polygonId}));
    if(!menu.isConnected||generation!==state.initGeneration)return;
    actions.replaceChildren();items=found.items;
    if(!items.length){
      actions.append(el('div',{class:'ctx-title',text:'Not in this review list'}),act('Suggest this polygon for review',async()=>{
        try{await postJSON('/nominate',{item:itemId,ids:[polygonId]});message('Sent to owner for approval.');}
        catch(error){message(error.message,true);}
      }));return;
    }
  }
  if(!items.length)return;
  const available=items.filter(item=>!['applied','applying','requested'].includes(item.state)),locked=items.length-available.length;
  actions.append(el('div',{class:'ctx-title',text:available.length===1?'Review · '+available[0].title:'Review · '+available.length+' editable items'}));
  if(locked)actions.append(el('div',{class:'ctx-title',text:locked+' completed or locked item'+(locked===1?' is':'s are')+' unchanged.'}));
  if(!state.reviewer)actions.append(act('Remove from review',()=>removeReviewItems(items.map(item=>item.id))));
  if(!available.length){actions.append(el('div',{class:'ctx-title',text:'Reopen completed items in Review to change them.'}));return;}
  for(const name of new Set(available.flatMap(item=>item.suggestions||[])))actions.append(act('Use “'+name+'”',()=>decideReviewItems(available,'relabel',name)));
  actions.append(act('Another label / comment…',()=>reviewLabelDialog(available)),
    act('Keep current label',()=>decideReviewItems(available,'keep')),act('Reject',()=>decideReviewItems(available,'reject')),
    act('Unable to decide',()=>decideReviewItems(available,'unable')));
}

async function startReviewer(access) {
  state.reviewer=true;state.apiBase=location.pathname.split('/window/')[0]+'/review-public/'+access;
  document.body.classList.add('standalone','reviewer-page');
  state.splitMarkerOptions={...SPLIT_DEFAULTS,open:false};
  const host=el('aside',{class:'reviewer-queue'}), title=el('h2',{text:'Review'}), content=el('div');
  host.append(title,content);document.body.prepend(host);
  reviewPanelLayout(host,title,access);
  const referrer=document.createElement('meta');referrer.name='referrer';referrer.content='no-referrer';document.head.append(referrer);
  // Review API disallows all research writes. Hide editing entrances as well.
  document.querySelector('#drawtools').hidden=true;
  window.SpacemapReview.mount(content,{publicMode:true});
}
window.SpacemapReview={mount:reviewQueue,flag:flagReview,start:startReviewer};


function reviewCellContext(){
  return state.layers.filter(l=>l.type==='cells'&&l.table).map(l=>({token:l.token,table:l.table}));
}

function reviewPanelLayout(host,title,access){
  let opt={left:340,right:380,leftOpen:true,rightOpen:true};
  const key='spacemap-review-panels-'+access.split('.')[0];
  try{Object.assign(opt,JSON.parse(localStorage.getItem(key)||'{}'));}catch(_){}
  const bar=el('div',{class:'reviewer-toolbar'}),layers=$('panel'),app=$('app');
  const leftButton=el('button',{class:'btn small','aria-label':'Toggle review panel',onclick:()=>{opt.leftOpen=!opt.leftOpen;apply();}});
  const rightButton=el('button',{class:'btn small','aria-label':'Toggle layers panel',onclick:()=>{opt.rightOpen=!opt.rightOpen;apply();}});
  bar.append(leftButton,el('strong',{text:'SPACEMAP Review'}),el('span',{class:'grow'}),rightButton);document.body.prepend(bar);
  const right=el('div',{class:'reviewer-layers'});layers.before(right);right.append(layers);
  const separator=(side)=>{
    const handle=el('div',{class:'reviewer-resizer '+side,role:'separator',tabindex:0,'aria-orientation':'vertical','aria-label':'Resize '+(side==='left'?'review':'layers')+' panel'});
    handle.onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();handle.setPointerCapture(e.pointerId);const start=e.clientX,width=opt[side];
      handle.onpointermove=v=>{opt[side]=width+(v.clientX-start)*(side==='left'?1:-1);apply();};
      handle.onpointerup=()=>{handle.onpointermove=null;handle.releasePointerCapture(e.pointerId);};};
    handle.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();opt[side]+=(e.key==='ArrowRight'?20:-20)*(side==='left'?1:-1);apply();}};
    return handle;
  };
  const leftResize=separator('left'),rightResize=separator('right');host.after(leftResize);right.before(rightResize);
  title.append(el('button',{class:'btn small',text:'«',title:'Hide review panel',onclick:()=>{opt.leftOpen=false;apply();}}));
  const close=el('button',{class:'btn small reviewer-close-layers',text:'»',title:'Hide layers panel',onclick:()=>{opt.rightOpen=false;apply();}});right.prepend(close);
  function apply(){
    for(const side of ['left','right'])opt[side]=Math.max(220,Math.min(innerWidth*.65,Number(opt[side])||340));
    document.body.style.setProperty('--review-left',opt.leftOpen?opt.left+'px':'0px');
    right.style.width=opt.right+'px';host.hidden=!opt.leftOpen;right.hidden=!opt.rightOpen;leftResize.hidden=!opt.leftOpen;rightResize.hidden=!opt.rightOpen;
    document.body.classList.toggle('review-left-hidden',!opt.leftOpen);
    leftButton.textContent=opt.leftOpen?'« Review':'» Show Review';rightButton.textContent=opt.rightOpen?'Layers »':'Show Layers «';
    try{localStorage.setItem(key,JSON.stringify(opt));}catch(_){}
    requestAnimationFrame(()=>{state.viewer?.forceRedraw();if(splitMarkers.panel)placeSplitMarkers();});
  }
  window.addEventListener('resize',apply);apply();
}

// One small notification request, independent of viewer rendering and job progress.
let reviewNotificationBusy=false,reviewNotificationSession='',reviewNotificationRevision=null;
setInterval(async()=>{
  const session=state.spec?.session||new URLSearchParams(location.search).get('session');
  if(state.reviewer||!state.studio||!session||reviewNotificationBusy)return;
  reviewNotificationBusy=true;
  try{
    if(reviewNotificationSession!==session){reviewNotificationSession=session;reviewNotificationRevision=null;}
    const found=await getJSON('/studio/'+encodeURIComponent(session)+'/review?action=notifications&since='+(reviewNotificationRevision||0));
    const header=$('studio-header')||document.querySelector('.studio-header');
    let badge=$('review-notification');
    if(!badge&&header){badge=el('button',{id:'review-notification',class:'btn small',onclick:()=>document.querySelector('[data-step=review]')?.click()});header.append(badge);}
    if(badge){badge.hidden=!found.pending;badge.textContent='Review · '+found.pending+' received';}
    if(reviewNotificationRevision!==null&&found.new?.length){message('Review update from '+[...new Set(found.new.map(e=>e.name))].join(', ')+'. Open Review to inspect.');document.dispatchEvent(new CustomEvent('spacemap-review-changed'));}
    reviewNotificationRevision=found.revision;
  }catch(_){}finally{reviewNotificationBusy=false;}
},3000);

window.SpacemapReview.flagResult = (job_id, groups, mode='') => flagReview({action:'flag_result',job_id,groups,mode});
window.SpacemapReview.flagCells = payload => flagReview({...payload,action:'flag_cells'});
