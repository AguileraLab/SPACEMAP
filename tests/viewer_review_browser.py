"""Real Firefox check: paged imported polygons, owner/reviewer decisions and grants."""
import json
from viewer_threshold_browser import main


def setup(root):
    from shapely.geometry import box, mapping
    import numpy as np
    import tifffile
    import pandas as pd
    from spacemap_gui.studio.phenotyping_inputs import save_folders
    # This fixture shares one exact image; the generic harness also makes a
    # background copy with the same ROI name for unrelated background tests.
    (root/'roi_bg_subtracted.ome.tiff').unlink()
    save_folders(root,{'images':str(root),'tables':str(root)})
    table=root/'cells.csv';frame=pd.read_csv(table);frame['timepoint']=1;frame['private_note']='not shared in image metadata';frame.to_csv(table,index=False)
    tifffile.imwrite(root/"roi.ome.tiff", np.full((5,400,400),1200,dtype=np.uint16), metadata={"axes":"CYX","Channel":{"Name":["M0","M1","HE_Red","HE_Green","HE_Blue"]}})
    features=[]
    for i in range(305):
        x,y=5+(i%20)*15,5+(i//20)*15
        features.append({'type':'Feature','geometry':mapping(box(x,y,x+8,y+8)),
                         'properties':{'id':f'polygon-{i:04d}','classification':{'name':'tumor','color':[255,0,0]}}})
    features.append({'type':'Feature','geometry':mapping(box(350,350,354,354)),
                     'properties':{'id':'normal-1','classification':{'name':'normal','color':[0,255,0]}}})
    (root/'tissue.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))


DRIVER=r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(f){await fetch('/result?progress='+encodeURIComponent(String(f)));for(let i=0;i<400;i++){if(f())return;await delay(100);}throw Error('Timed out '+String(f)+' '+[...document.querySelectorAll('dialog[open],.review-status')].map(n=>n.innerText).join(' | ')+document.body.innerText.slice(-400));}
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;};
const btn=(name,scope=document)=>Array.from(scope.querySelectorAll('button')).find(b=>b.textContent===name&&b.getClientRects().length);
window.prompt=()=> 'normal';window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.error?.stack)));
if(!new URLSearchParams(location.search).has('review'))setTimeout(async()=>{try{
 const state=window.__state;
 await until(()=>state.studio&&state.viewer&&state.layers.some(l=>l.type==='annotations'));
 const image=state.layers.find(l=>l.type==='image'),annotation=state.layers.find(l=>l.type==='annotations');
 await until(()=>image.item?.getFullyLoaded());
 const originalImageItem=image.item;
 setVisible(annotation,true);
 document.querySelector('[data-layer-group=annotations] > summary').click();
 await until(()=>document.querySelector('[data-annotation-source="Tissue"]'));
 await delay(1200);
 let source=document.querySelector('[data-annotation-source="Tissue"]');
 for(const parent of [source.parentElement,source])if(parent.matches('details')&&!parent.open)parent.querySelector('summary').click();
 await until(()=>document.querySelector('[data-annotation-source="Tissue"] [data-component-class=tumor]'));
 source=document.querySelector('[data-annotation-source="Tissue"]');
 source.querySelector('[data-component-class=tumor] > summary').click();
 await until(()=>document.querySelector('[data-annotation-source="Tissue"]').querySelectorAll('[data-component-id]').length===100);
 source=document.querySelector('[data-annotation-source="Tissue"]');
 assert(source.querySelectorAll('[data-component-id]').length===100,'Only one hundred rows render');
 const summary=source.querySelector('[data-component-class=tumor] > summary'),colour=summary.querySelector('.component-colour');
 assert(colour.getBoundingClientRect().width>=20&&getComputedStyle(colour).backgroundColor==='rgb(255, 0, 0)','Original colour visible at checkbox height');
 assert(Math.abs(summary.querySelector('input').getBoundingClientRect().top-colour.getBoundingClientRect().top)<8,'Disclosure, checkbox and colour share one row');
 assert(!source.querySelector('input[type=search]'),'No redundant polygon search');
 assert(source.querySelector('summary .component-order'),'Sort is beside the annotation name');
 assert(source.querySelector('[data-component-id] .component-colour'),'Every polygon keeps its colour swatch');
 assert(summary.querySelector('.component-count').textContent==='305 polygons','Count identifies individual polygons');
 assert(!btn('Group',source)&&!btn('Ungroup',source),'No redundant display grouping buttons');
 const appearance=()=>document.querySelector('[aria-label="Annotation appearance"]');
 assert(appearance()?.getClientRects().length,'Annotation appearance is in the expanded header');
 const opacity=()=>document.querySelector('[aria-label="Annotation opacity"]');
 assert(document.querySelectorAll('[data-layer-group=annotations] input[type=range]').length===1,'Only one opacity slider exists for all annotations');
 assert(document.querySelector('[data-layer-group=annotations]').children[1].contains(opacity()),'Shared opacity is immediately below the Annotations header');
 const imageOpacity=image.item.getOpacity();opacity().value='27';opacity().dispatchEvent(new Event('input'));
 await until(()=>annotation.item);
 assert(annotation.item.getOpacity()===0.27&&image.item.getOpacity()===imageOpacity,'Universal annotation opacity changes the overlay, not the image');
 saveView();assert(JSON.parse(localStorage.getItem('spacemap-viewer-view:'+state.spec.id)).annotationOpacity===0.27,'Shared opacity is saved');
 opacity().value='100';opacity().dispatchEvent(new Event('input'));
 appearance().value='fill';appearance().dispatchEvent(new Event('change'));
 assert(state.annotationMode==='fill','Filled appearance updates viewer state');
 const checkAlpha=async(mode)=>{const blob=await fetch(api('/annotations/'+annotation.token+'/tile/0/0/0.png?mode='+mode)).then(r=>r.blob());const bitmap=await createImageBitmap(blob);const c=document.createElement('canvas');c.width=bitmap.width;c.height=bitmap.height;c.getContext('2d').drawImage(bitmap,0,0);return c.getContext('2d').getImageData(9,9,1,1).data[3];};
 assert(await checkAlpha('fill')===255&&await checkAlpha('outline')===0,'Imported polygon fill is solid at 100%, with transparent interiors in outline mode');
 appearance().value='both';appearance().dispatchEvent(new Event('change'));
 assert([...source.querySelectorAll('[data-component-id] .grow')].every(e=>e.textContent==='tumor'),'Component names use the parent class');
 await pickAnnotation({x:9,y:9},false);await until(()=>annotation.annotationOutlineSvg?.isConnected);
 assert(annotationSelection(annotation).has('polygon-0000'),'Viewer click selects exact component');
 source=document.querySelector('[data-annotation-source="Tissue"]');await until(()=>{source=document.querySelector('[data-annotation-source="Tissue"]');return source?.querySelector('[data-component-id="polygon-0000"] input');});
 assert(source.querySelector('[data-component-id="polygon-0000"] input').checked,'Viewer selection checks Layers row');
 await pickAnnotation({x:24,y:9},true);
 assert(annotationSelection(annotation).size===2,'Ctrl click preserves multiple selected polygons');
 annotation.hidden.add('tumor');bump(annotation);await pickAnnotation({x:9,y:9},false);
 assert(annotationSelection(annotation).size===0,'Hidden polygons cannot be selected');
 annotation.hidden.clear();bump(annotation);await delay(600);
 source=document.querySelector('[data-annotation-source="Tissue"]');
 await until(()=>source.querySelector('[data-component-id]'));
 const other={id:'other-source',name:'Second annotation source',type:'shapes',visible:true,opacity:1,review_inline:false,regions:true,
   image_token:image.token,features:[],local:true,editable:true};state.layers.push(other);
 other.features=[{type:'Feature',geometry:{type:'Polygon',coordinates:[[[330,5],[338,5],[338,13],[330,13],[330,5]]]},properties:{id:'other-polygon',classification:{name:'normal',color:[0,255,0]}}}];
 renderShapes();await pickAnnotation({x:9,y:9},false);await pickAnnotation({x:334,y:9},true);
 assert(annotationSelection(annotation).size===1&&annotationSelection(other).size===1,'Selection spans annotation classes and sources');
 setVisible(other,false);await pickAnnotation({x:334,y:9},false);assert(annotationSelectedGroups().length===0,'Hidden source excluded from hit testing');
 setVisible(other,true);
 document.querySelector('[data-mode=select]').click();
 const gesture=async(win,tool,points)=>{
   win.document.querySelector('[data-tool='+tool+']').click();
   const viewer=win.__state.viewer,pos=p=>win.screenOf({x:p[0],y:p[1]});
   if(tool==='polygon'){
     for(const p of points)viewer.raiseEvent('canvas-click',{position:pos(p),quick:true,shift:false,originalEvent:{}});
     viewer.raiseEvent('canvas-double-click',{position:pos(points[points.length-1])});
   }else{
     viewer.raiseEvent('canvas-press',{position:pos(points[0]),originalEvent:{button:0}});
     for(const p of points.slice(1))viewer.raiseEvent('canvas-drag',{position:pos(p),delta:new win.OpenSeadragon.Point(0,0)});
     viewer.raiseEvent('canvas-release',{position:pos(points[points.length-1])});
   }
 };
 await gesture(window,'rect',[[0,0],[345,15]]);
 await until(()=>annotationSelection(annotation).size===20&&annotationSelection(other).size===1);
 assert(annotationSelection(annotation).size===20&&annotationSelection(other).size===1,'Rectangle selects fully enclosed polygons across visible classes and sources');
 annotation.hidden.add('tumor');bump(annotation);
 await gesture(window,'polygon',[[325,0],[345,0],[345,15],[325,15]]);
 await until(()=>annotationSelection(annotation).size===0&&annotationSelection(other).size===1);
 assert(annotationSelection(annotation).size===0,'Polygon tool excludes hidden classes');
 annotation.hidden.clear();bump(annotation);
 await gesture(window,'freehand',[[0,0],[30,0],[30,15],[0,15],[0,0]]);
 await until(()=>annotationSelection(annotation).size===2&&annotationSelection(other).size===0);
 assert(annotationSelection(annotation).size===2,'Freehand selection picks exact enclosed polygons');
 document.querySelector('[data-tool=off]').click();
 other.features=Array.from({length:2000},(_,i)=>{const x=5+i%50*6,y=5+Math.floor(i/50)*6;return {type:'Feature',geometry:{type:'Polygon',coordinates:[[[x,y],[x+3,y],[x+3,y+3],[x,y+3],[x,y]]]},properties:{id:'local-'+i,classification:{name:'normal',color:[0,255,0]}}};});
 const exact=JSON.stringify(other.features);renderShapes();
 const paths=()=>state.svgRoot.querySelectorAll('[data-layer="other-source"] path').length;
 assert(paths()===1,'Two thousand local polygons use one compound display path');
 (state.annotationRowsOpen||(state.annotationRowsOpen={}))[annotationRowKey(other,'normal')]=true;const start=performance.now();renderShapes();const expandedMs=performance.now()-start;
 assert(paths()===1&&JSON.stringify(other.features)===exact,'Opening a local polygon list keeps rendering bounded and geometry unchanged');
 await fetch('/result?progress='+encodeURIComponent('Local 2000-polygon expanded redraw: '+expandedMs.toFixed(1)+' ms; one SVG path'));
 removeLayer(other,true);clearAnnotationSelection();state.componentLists[annotation.token].classes.tumor.offset=0;buildPanel();await delay(500);source=document.querySelector('[data-annotation-source="Tissue"]');await until(()=>source.querySelectorAll('[data-component-id]').length===100);

 btn('All',source).click();await until(()=>source.textContent.includes('305 selected'));
 assert(source.querySelectorAll('[data-component-id]').length===100,'Selecting all does not render all polygons');
 btn('None',source).click();await until(()=>source.textContent.includes('0 selected'));
 source.querySelector('[data-component-id] input').click();
 btn('Flag for review',source).click();
 await until(()=>state.lastFlaggedReview?.length===1);
 assert(!document.querySelector('dialog[open]'),'Flagging needs no popup');
 btn('Suggestions…',document.querySelector('#message')).click();
 await until(()=>document.querySelector('dialog[open] [aria-label="Suggest tumor"]'));
 let modal=document.querySelector('dialog[open]');
 assert(modal.querySelector('[aria-label="Suggest Other"]'),'Existing classes and Other are available');
 modal.querySelector('[aria-label="Suggest tumor"]').click();
 modal.querySelector('[aria-label="Custom suggestion"]').value='normal';btn('Add',modal).click();
 btn('Apply to 1 item',modal).click();await until(()=>!document.querySelector('dialog[open]'));
 const route='/studio/browser-check/review';let queue=await getJSON(route);
 assert(queue.total===1,'One exact polygon flagged');assert(queue.items[0].suggestions.length===2,'Suggestions saved');
 const existing=queue.items[0].target.ids[0], extra=existing==='polygon-0001'?'polygon-0002':'polygon-0001';
 selectAnnotations(annotation,[extra],{replace:true});await flagSelectedAnnotations();queue=await getJSON(route);
 assert(queue.total===2&&queue.items[1].suggestions.join(',')==='tumor,normal','Next flag remembers choices without opening a dialog');
 assert(!document.querySelector('dialog[open]'),'Repeated flags stay quick');
 const nav=document.querySelector('[data-step=review]');assert(nav,'Review is a Studio step');nav.click();
 await until(()=>document.querySelector('.review-list .review-table tbody tr'));
 await until(()=>document.querySelector('.review-images tbody tr'));
 const imageTable=document.querySelector('.review-images table'), scroller=document.querySelector('.review-images'), divider=imageTable.querySelectorAll('[role=separator]')[1];
 assert(imageTable.tBodies[0].rows.length>0,'Loaded images appear in Review');
 assert(![...imageTable.querySelectorAll('th')].some(th=>th.textContent==='Cells')&&imageTable.textContent.includes('Cell labels'),'No empty cell-count column; Cell labels is clear');
 assert(scroller.scrollWidth>scroller.clientWidth,'Review image columns scroll horizontally instead of squeezing names');
 divider.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));const wider=parseFloat(imageTable.querySelectorAll('col')[1].style.width);
 assert(wider>235,'Review column divider resizes its column');
 divider.dispatchEvent(new MouseEvent('dblclick',{bubbles:true}));
 assert(state.reviewWidths.images.length===imageTable.querySelectorAll('th').length,'Double-click fits every column');
 document.querySelector('.review-list [aria-label="Select this page"]').click();
 await until(()=>document.querySelector('.review-list [aria-label="Select this page"]').checked);
 document.querySelector('.review-list [aria-label="Select this page"]').click();
 assert([...document.querySelectorAll('.review-list tbody input[type=checkbox]')].every(t=>!t.checked),'Second header click deselects the page');
 document.querySelector('.review-list tbody input[type=checkbox]').click();
 assert(document.querySelector('.review-list [aria-label="Select this page"]').indeterminate,'Header shows partial selection immediately');
 document.querySelector('.review-list [aria-label="Select this page"]').click();
 assert(document.querySelector('.review-list [aria-label="Select this page"]').checked,'Header shows all selected immediately');
 btn('Suggestions…',document.querySelector('#studio-tools')).click();
 await until(()=>document.querySelector('dialog[open] [aria-label="Suggest Other"]'));
 modal=document.querySelector('dialog[open]');modal.querySelector('[aria-label="Suggest Other"]').click();
 btn('Apply to 2 items',modal).click();await until(()=>!document.querySelector('dialog[open]'));
 queue=await getJSON(route);
 assert(queue.items.every(i=>i.suggestions.includes('Other')),'Choices can be applied to multiple flagged polygons later');
 document.querySelector('.review-list .review-table tbody tr').click();await until(()=>state.layers.some(l=>l.id==='review-focus'));
 assert(state.layers.find(l=>l.type==='image').item===originalImageItem,'Selecting review in current image preserves the image');
 await until(()=>state.layers.find(l=>l.id==='review-focus').features.length===2);
 const reviewSection=document.querySelector('[data-annotation-section="Review polygons"]'),imports=document.querySelector('[data-annotation-root=imports]');
 assert(!!(imports.compareDocumentPosition(reviewSection)&Node.DOCUMENT_POSITION_FOLLOWING),'Review polygons follows Imported GeoJSON');
 assert(reviewSection.querySelectorAll('[data-review-polygon]').length===2,'Review polygons lists all flagged polygons, not just the active one');
 assert(reviewSection.classList.contains('annotation-subgroup')&&getComputedStyle(reviewSection).borderRadius===getComputedStyle(imports).borderRadius,'Review has the same bordered panel as Imports');
 assert(Math.abs(reviewSection.getBoundingClientRect().left-imports.getBoundingClientRect().left)<1,'Review and Imported panel arrows align');
 assert(reviewSection.querySelectorAll('[aria-label="Another review label"]').length===2&&reviewSection.querySelector('[aria-label="Use normal for this review item"]'),'Review rows have inline suggestions and an extra-label button');
 opacity().value='42';opacity().dispatchEvent(new Event('input'));
 const working=state.layers.find(l=>l.id==='other-source');
 assert(state.layers.filter(isAnnotationDisplayLayer).every(l=>l.opacity===0.42),'Shared slider includes Working, Imported, Review and hidden polygons');
 assert(state.svgRoot.querySelector('[data-layer="review-focus"]').getAttribute('opacity')==='0.42','Review polygons render with the same opacity');
 const late={id:'late-review-fixture',type:'shapes',visible:true,opacity:1,features:[],review_inline:true};state.layers.push(late);renderShapes();
 assert(late.opacity===0.42,'Newly arriving review polygons inherit the shared opacity');state.layers.splice(state.layers.indexOf(late),1);
 opacity().value='100';opacity().dispatchEvent(new Event('input'));
 assert(parseFloat(getComputedStyle(reviewSection.querySelector('.review-polygon-list')).marginLeft)>0,'Review polygon rows are indented');
 btn('Clear selection').click();
 await until(()=>!document.querySelector('.review-list [aria-label="Select this page"]').checked&&annotationSelectedGroups().length===0);
 assert(state.layers.find(l=>l.id==='review-focus').features.length===2,'Clearing selection keeps the complete Review polygons list');
 const previewFeatures=state.layers.find(l=>l.id==='review-focus').features;
 document.querySelector('[data-review-polygon] input').click();
 await until(()=>document.querySelector('.review-list [aria-label="Select this page"]').indeterminate);
 assert(JSON.stringify(state.layers.find(l=>l.id==='review-focus').features)===JSON.stringify(previewFeatures),'Selection preserves exact loaded review geometry');
 const chosenPreview=state.layers.find(l=>l.id==='review-focus'),chosenId=[...annotationSelection(chosenPreview)][0],chosenFeature=chosenPreview.features.find(f=>f.properties.id===chosenId);
 const linkedSource=state.layers.find(l=>l.token===chosenFeature.properties.review_source);
 assert(annotationSelection(linkedSource).has(chosenFeature.properties.review_polygon),'Review Layers checkbox selects the original source polygon too');
 selectAnnotations(linkedSource,[chosenFeature.properties.review_polygon],{toggle:true});
 await until(()=>annotationSelection(chosenPreview).size===0);
 assert(!document.querySelector('.review-list [aria-label="Select this page"]').checked,'Toggling source selection also clears the Review list');
 btn('Select all matching').click();await until(()=>document.querySelector('.review-list [aria-label="Select this page"]').checked);
 assert(annotationSelection(chosenPreview).size===2,'Select all matching includes every review polygon');
 btn('Metadata columns…').click();await until(()=>document.querySelector('[aria-label="Find metadata columns"]'));
 let metadataDialog=document.querySelector('dialog[open]');const timepointLabel=[...metadataDialog.querySelectorAll('label')].find(l=>l.textContent==='timepoint');timepointLabel.querySelector('input').click();btn('Apply columns',metadataDialog).click();
 await until(()=>document.querySelector('.review-images th')&&[...document.querySelectorAll('.review-images th')].some(th=>th.textContent==='timepoint'));
 assert(document.querySelector('.review-images tbody').textContent.includes('1'),'Image table shows selected metadata');
 const item=queue.items[0];
 const share=await postJSON(route,{action:'share',name:'Pathologist test',items:[item.id]});
 btn('Share…').click();await until(()=>document.querySelector('dialog[open] [aria-label="Saved reviewer link"] option[value="'+share.id+'"]'));
 modal=document.querySelector('dialog[open]');const savedSelect=modal.querySelector('[aria-label="Saved reviewer link"]');savedSelect.value=share.id;savedSelect.dispatchEvent(new Event('change'));
 await until(()=>modal.querySelector('[aria-label="Review link"]')?.value===share.url);
 assert(modal.querySelector('[aria-label="Review link"]').readOnly,'Choosing a saved reviewer automatically displays the saved URL');
 assert(modal.querySelector('[aria-label="Review link name"]').value==='Pathologist test','Share menu lists and loads saved labelled links');
 modal.querySelector('label input[type=checkbox]').checked=false;
 btn('Update link',modal).click();await until(()=>modal.querySelector('[aria-label="Review link"]')&&!btn('Update link',modal)?.disabled&&!btn('Create link',modal)?.disabled);
 assert(modal.querySelector('[aria-label="Review link"]').value===share.url,'Editing a saved link preserves its URL and existing review list');
 btn('Close',modal).click();
 const access=share.url.split('/review-public/')[1].replace(/\/$/,'');
 const base='/review-public/'+access;
 await postJSON(route,{action:'destination',id:share.id});
 await postJSON(route,{action:'add_to_link',id:share.id,items:[item.id]});
 const external=await getJSON(base+'/queue');
 assert(!JSON.stringify(external).includes('source'),'Review queue does not expose source paths');
 const response=await fetch(api(base+'/studio/browser-check/review'),{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
 assert(response.status===403,'Link cannot reach owner mutation APIs');
 const spec=await getJSON(base+'/view?id='+item.id);
 assert(!spec.spec.session&&!spec.spec.workspace,'Reviewer view has no owner session or workspace');
 const publicImage=spec.spec.layers.find(l=>l.type==='image').token;
 const pixels=await fetch(api(base+'/image/'+publicImage+'/info'));assert(pixels.ok,'Shared image can be viewed');
 const ownerTokenDenied=await fetch(api('/image/'+publicImage+'/info'));assert(ownerTokenDenied.status===404,'Reviewer aliases cannot access the ordinary owner API');
 const denied=await fetch(api(base+'/image/not-shared/info'));assert(denied.status===403,'Unshared image blocked');
 const frame=window.open(share.url,'review-fixture','width=1400,height=900');assert(frame,'Reviewer window opened');
 await until(()=>frame.document?.querySelector('.review-list tbody tr'));
 frame.document.querySelector('.review-list tbody tr').click();
 await until(()=>frame.__state?.imagesReady&&frame.__state?.layers.some(l=>l.type==='annotations'&&l.item));
 const reviewerState=frame.__state,reviewerSource=reviewerState.layers.find(l=>l.type==='annotations');
 await until(()=>reviewerSource.annotationOutlineSvg?.isConnected||reviewerSource.annotationSelectionOverlay?.item?.getFullyLoaded());
 assert(reviewerState.reviewer&&reviewerSource.visible,'Reviewer shares the tiled context and selected polygon outline');
 assert(!frame.document.querySelector('#drawtools').hidden&&!frame.document.querySelector('[data-tool=rect]').disabled,'Reviewer has area-selection tools');
 assert(frame.document.querySelector('[aria-label="Annotation appearance"]'),'Reviewer has the same annotation appearance control');
 const reviewerOpacity=frame.document.querySelector('[aria-label="Annotation opacity"]');reviewerOpacity.value='35';reviewerOpacity.dispatchEvent(new frame.Event('input'));
 assert(reviewerState.layers.filter(frame.isAnnotationDisplayLayer).every(l=>l.opacity===0.35)&&state.annotationOpacity===1,'Reviewer opacity is universal and independent of the owner');
 reviewerOpacity.value='100';reviewerOpacity.dispatchEvent(new frame.Event('input'));
 assert(frame.document.querySelector('[data-tool=merge]').hidden&&frame.document.querySelector('[data-tool=split]').hidden,'Reviewer cannot merge or split source polygons');

 assert(frame.document.querySelector('[data-annotation-section="Review polygons"]'),'Review focus has its own annotation section');
 const he=reviewerState.layers.find(l=>l.id==='he');assert(he,'Registered H&E is shared');
 const resize=frame.document.querySelector('[aria-label="Resize layers panel"]'),panel=frame.document.querySelector('.reviewer-layers');
 const oldWidth=panel.getBoundingClientRect().width;resize.dispatchEvent(new frame.KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}));
 assert(panel.getBoundingClientRect().width>oldWidth,'Reviewer layers can resize');
 frame.document.querySelector('[aria-label="Toggle layers panel"]').click();assert(panel.hidden,'Layers collapse');
 frame.document.querySelector('[aria-label="Toggle layers panel"]').click();assert(!panel.hidden,'Layers reopen from visible toolbar');
 frame.document.querySelector('[aria-label="Toggle review panel"]').click();assert(frame.document.querySelector('.reviewer-queue').hidden,'Review list collapses');
 frame.document.querySelector('[aria-label="Toggle review panel"]').click();
 const beforeItem=reviewerState.layers.find(l=>l.type==='image').item;
 const newFlag=await postJSON(route,{action:'flag',token:annotation.token,ids:['polygon-0010'],cell_context:reviewCellContext()});
 await until(()=>frame.document.querySelectorAll('.review-list tbody tr').length===2);
 assert(reviewerState.layers.find(l=>l.type==='image').item===beforeItem,'New flags arrive on same link without reloading viewer');
 frame.document.querySelector('.review-list [aria-label="Select this page"]').click();
 await until(()=>reviewerState.layers.find(l=>l.id==='review-focus')?.features.length===2);
 frame.setVisible(reviewerSource,false);
 const preview=reviewerState.layers.find(l=>l.id==='review-focus');
 assert(preview.visible&&preview.features.length===2,'All checked review polygons remain independently visible when source family is hidden');
 assert(frame.document.querySelectorAll('[data-review-polygon]').length===2,'Working annotations lists both checked review polygons');
 const previewPath=reviewerState.svgRoot.querySelector('[data-layer="review-focus"] path');
 assert(previewPath&&previewPath.getBoundingClientRect().width>0,'Review selection has visible rendered polygons');
 const reviewChoices=()=>frame.document.querySelector('[data-review-polygon] .review-polygon-choices');
 assert(reviewChoices()?.querySelector('[aria-label="Use normal for this review item"]'),'Reviewer Layers offers the shared suggestions');
 reviewChoices().querySelector('[aria-label="Another review label"]').click();
 await until(()=>frame.document.querySelector('dialog[open] .review-label-options button'));
 let labelDialog=frame.document.querySelector('dialog[open]');
 assert([...labelDialog.querySelectorAll('.review-label-options button')].some(b=>b.textContent==='tumor'),'Plus popup offers existing annotation names');
 labelDialog.querySelector('[aria-label="Review label"]').value='New reviewer label';btn('Use label',labelDialog).click();
 await until(()=>!frame.document.querySelector('dialog[open]')&&preview.features.some(f=>f.properties.classification.name==='New reviewer label'));
 assert(preview.features.some(f=>f.properties.classification.name==='New reviewer label'),'Plus popup accepts a new label without editing source data');
 assert(preview.features.every(f=>f.properties.classification.name==='New reviewer label'),'Plus applies the new label to every selected review polygon');
 const inlineRow=[...frame.document.querySelectorAll('[data-review-polygon]')].find(row=>row.textContent.includes('New reviewer label'));
 inlineRow.querySelector('[aria-label="Use normal for this review item"]').click();
 await until(()=>!preview.features.some(f=>f.properties.classification.name==='New reviewer label'));
 assert((await getJSON(base+'/queue')).items.every(i=>i.decision?.label==='normal'),'Inline suggestion on a selected polygon applies to the full selection');
 const onePreview=preview.features[0],twoPreview=preview.features[1];
 frame.selectAnnotations(preview,[onePreview.properties.id],{replace:true});
 const unselectedRow=frame.document.querySelector('[data-review-polygon="'+twoPreview.properties.review_item+'"]');
 unselectedRow.querySelector('[aria-label="Another review label"]').click();
 await until(()=>frame.document.querySelector('dialog[open] [aria-label="Review label"]'));
 labelDialog=frame.document.querySelector('dialog[open]');labelDialog.querySelector('[aria-label="Review label"]').value='Only unselected polygon';btn('Use label',labelDialog).click();
 await until(()=>!frame.document.querySelector('dialog[open]')&&preview.features.some(f=>f.properties.classification.name==='Only unselected polygon'));
 assert(preview.features.find(f=>f.properties.id===onePreview.properties.id).properties.classification.name==='normal','Suggestion on an unselected row leaves selected polygons untouched');
 frame.selectAnnotations(preview,preview.features.map(f=>f.properties.id),{replace:true});
 const targetFeature=preview.features.find(f=>f.properties.review_item===item.id),coords=targetFeature.geometry.coordinates[0];
 const mx=(coords[0][0]+coords[2][0])/2,my=(coords[0][1]+coords[2][1])/2;
 frame.document.querySelector('[data-tool=off]').click();
 const screen=frame.screenOf({x:mx,y:my}),bounds=reviewerState.viewer.element.getBoundingClientRect();
 reviewerState.viewer.canvas.dispatchEvent(new frame.MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:bounds.left+screen.x,clientY:bounds.top+screen.y,button:2}));
 await until(()=>btn('Use “normal”',frame.document));btn('Use “normal”',frame.document).click();
 await until(()=>preview.features.find(f=>f.properties.review_item===item.id)?.properties.classification.name==='normal');
 assert(preview.features.find(f=>f.properties.review_item===item.id).properties.colour==='#00ff00','Right-click suggestion previews destination family colour');
 const currentShared=await getJSON(base+'/queue');assert(currentShared.items.length===2&&currentShared.items.every(i=>i.decision?.label==='normal'),'Right-click decision applies to all selected review polygons and reaches the owner');
 frame.setVisible(preview,false);await delay(3300);assert(!preview.visible,'Background review polling preserves explicitly hidden previews');
 frame.setVisible(preview,true);
 const rc=reviewerState.layers.find(l=>l.type==='cells');frame.setVisible(rc,true);
 await until(()=>rc.item?.getFullyLoaded());
 const cellGroup=frame.document.querySelector('[data-layer-group=cells]');if(!cellGroup.open)cellGroup.querySelector('summary').click();
 const cellCard=()=>frame.document.querySelector('[data-layer-id="'+rc.id+'"]');
 if(cellCard()?.classList.contains('collapsed'))cellCard().querySelector('.layer-head').click();
 const colourSelect=()=>[...frame.document.querySelectorAll('[data-layer-group=cells] select')].find(e=>[...e.options].some(o=>o.value==='phenotype_ref1'));
 await until(()=>colourSelect());colourSelect().value='phenotype_ref1';colourSelect().dispatchEvent(new frame.Event('change'));
 await until(()=>frame.document.querySelector('[aria-label="Colour for A"]'));
 const swatch=frame.document.querySelector('[aria-label="Colour for A"]');swatch.value='#123456';swatch.dispatchEvent(new frame.Event('change'));
 assert(rc.groupColours[JSON.stringify([rc.table,'phenotype_ref1'])].A==='#123456'&&!rc.hidden.has('A'),'Reviewer can recolour a category without hiding it');
 assert(!state.layers.find(l=>l.type==='cells').groupColours,'Reviewer colours do not modify owner display settings');
 const colInfo=await frame.getJSON('/cells/'+rc.token+'/columns?t='+rc.table);assert(colInfo.columns.includes('M0'),'Shared columns endpoint permits only bound table');
 rc.hidden.add('B');frame.bump(rc);
 frame.document.querySelector('[data-tool=rect]').click();const targetSelect=frame.document.querySelector('#area-select-target');targetSelect.value='cells';targetSelect.dispatchEvent(new frame.Event('change'));
 await gesture(frame,'rect',[[0,0],[400,400]]);
 await until(()=>reviewerState.cellInspectionSelection?.members.size===20);
 assert([...reviewerState.cellInspectionSelection.members.keys()].every(n=>n<=20),'Cell area selection excludes hidden classes');
 await until(()=>reviewerState.cellAreaOverlay?.item?.getFullyLoaded());
 assert(reviewerState.cellAreaOverlay.item.getFullyLoaded(),'Selected cells have a rendered tiled border');
 const ownerDataSource=reviewCellContext()[0];const unsharedSelection=await fetch(api(base+'/select_cells'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:ownerDataSource.token,points:[[0,0],[400,0],[400,400],[0,400]]})});assert(unsharedSelection.status===403,'Cell selection rejects owner tokens outside reviewer scope');
 frame.document.querySelector('[data-tool=off]').click();
 await frame.bringFiles([new frame.File(['x,y\n1,2'],'local.csv',{type:'text/csv'})]);
 assert(frame.document.querySelector('#message').textContent.includes('imports are unavailable'),'Reviewer imports cannot upload to owner storage');

 frame.openSplitTab('data');
 await until(()=>frame.document.querySelector('.data-grid tbody tr'));
 assert(frame.eval("dataView").result.total===40,'Reviewer Data shows matching image cells');
 assert(frame.document.querySelector('.data-head').textContent.includes('Read-only'),'Data is clearly read-only');
 assert(!btn('Save',frame.document.querySelector('.data-head')),'No write controls in reviewer Data');
 const sharedSource=frame.eval("dataView").source;
 const forbidden=await fetch(api(base+'/data/mutate'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:sharedSource,operation:{kind:'delete_rows',scope:'all'}})});
 assert(forbidden.status===403,'Server rejects table edits');
 const wide=await postJSON(base+'/data/query',{source:sharedSource,rows:{this_image:false},limit:200});assert(wide.total===40,'Caller cannot escape shared image scope');
 frame.floatSplitTab();await until(()=>frame.viewerPanelWindow('data'));
 assert(frame.viewerPanelWindow('data').document.querySelector('.data-grid tbody tr'),'Data floats with rendered rows');
 frame.viewerPanelWindow('data').close();
 frame.openSplitTab('markers');frame.floatSplitTab();await until(()=>frame.viewerPanelWindow('markers'));
 assert(frame.viewerPanelWindow('markers').document.querySelector('.split-marker-grid'),'Marker tiles float');
 frame.viewerPanelWindow('markers').close();frame.close();
 const latest=(await getJSON(base+'/queue')).items.find(i=>i.id===item.id);
 await postJSON(base+'/decision',{id:item.id,version:latest.version,decision:'relabel',label:'normal',note:'Confirmed from reviewer'});
 await until(()=>document.querySelector('.review-detail')?.textContent.includes('Confirmed from reviewer'));
 assert(state.layers.find(l=>l.type==='image').item===originalImageItem,'Live review polling leaves viewer intact');
 btn('Accept & apply').click();await until(()=>document.querySelector('.review-detail')?.textContent.includes('Applied'));
 queue=await getJSON(route);assert(queue.items[0].state==='applied','Owner approval applied');
 await postJSON(route,{action:'revoke',id:share.id});
 const closed=await fetch(api(base+'/queue'));assert(closed.status===403,'Revoked link immediately closed');
 btn('Clear selection').click();document.querySelector('.review-images tbody input[type=checkbox]').click();btn('Share…').click();
 await until(()=>document.querySelector('dialog[open] [aria-label="Review link name"]'));
 modal=document.querySelector('dialog[open]');modal.querySelector('[aria-label="Review link name"]').value='Image collaborator';
 btn('Shared metadata: none',modal).click();await until(()=>document.querySelector('[aria-label="Find metadata columns"]'));
 metadataDialog=[...document.querySelectorAll('dialog[open]')].at(-1);[...metadataDialog.querySelectorAll('label')].find(l=>l.textContent==='timepoint').querySelector('input').click();btn('Apply columns',metadataDialog).click();
 await until(()=>btn('Shared metadata: 1',modal));btn('Create link',modal).click();await until(()=>modal.querySelector('[aria-label="Review link"]')&&!btn('Update link',modal)?.disabled&&!btn('Create link',modal)?.disabled);
 const collaboratorURL=modal.querySelector('[aria-label="Review link"]').value;btn('Close',modal).click();
 const collaborator=window.open(collaboratorURL,'image-collaborator','width=1400,height=900');await until(()=>collaborator.document?.querySelector('.review-images tbody tr'));
 assert([...collaborator.document.querySelectorAll('.review-images th')].some(th=>th.textContent==='timepoint'),'Shared metadata is shown to collaborators');
 assert(!collaborator.document.querySelector('.review-images').textContent.includes('private_note'),'Unchosen metadata not shared');
 assert(!collaborator.document.querySelector('.review-list tbody tr'),'Image sharing does not manufacture a review item');
 collaborator.document.querySelector('.review-images tbody tr').click();await until(()=>collaborator.__state.imagesReady);
 assert(collaborator.__state.layers.some(l=>l.id==='he'),'Image-only collaborator can view image and H&E');collaborator.close();
 document.title='Review checks '+checks;
 await fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){await fetch('/result?error='+encodeURIComponent(e.message+' '+e.stack));}},150);
</script>'''

if __name__=='__main__':
    main(DRIVER,studio=True,setup=setup,timeout=240,allow_popups=True,screenshot=False)
