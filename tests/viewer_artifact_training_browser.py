"""Draw, edit, retrain and compare artifact models using the real viewer tools."""
from viewer_threshold_browser import main
from viewer_artifact_review_browser import setup as base_setup


def setup(root):
    import shutil
    from spacemap_gui.studio import annotation, artifact_learning
    base_setup(root)
    shutil.copyfile(root / 'markers.ome.tiff', root / 'second.ome.tiff')
    output = root / 'outputs/processed_annotations/stage0_tissue_annotation'
    shutil.copyfile(annotation.valid_tissue(output, 'markers'), annotation.valid_tissue(output, 'second'))
    artifact_learning.polygons_path(output, 'markers', 'clot').unlink()

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<2400;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 const read=p=>fetch(api(p)).then(r=>r.json());
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.some(l=>l.id==='tissue-regions'&&l.revision));
 const tissue=s.layers.find(l=>l.id==='tissue-regions'),original=(await read('/shapes/'+tissue.token)).revision;
 const originalFetch=window.fetch;
 window.fetch=(url,options)=>String(url).endsWith('/open_examples')?Promise.resolve(new Response(JSON.stringify({error:'Unknown artifact action.'}),{status:400,headers:{'Content-Type':'application/json'}})):originalFetch(url,options);
 button(document.querySelector('.studio-stage-tabs'),'Artifacts').click();
 await until(()=>tools().querySelector('[data-example-role=positive]'));
 assert(!tools().querySelector('[aria-label="Pinker by"]')&&!button(tools(),'Preview'),'Blood clots uses model prediction without the pinkness slider or detector Preview');
 const ex=()=>s.layers.find(l=>l.id==='artifact-examples-clot');
 const red=()=>s.layers.find(l=>l.id==='artifact-clot');
 const settled=()=>[ex(),red()].every(l=>!l?.savePending&&!l?.saveBusy&&!l?.brushBusy);
 const mark=async role=>{tools().querySelector('[data-example-role='+role+']').click();await until(()=>ex()?.revision&&drawTarget()===(role==='positive'?red():ex())).catch(()=>{
   throw Error('Class switch did not settle: role='+role+' target='+(s.drawInto||'none')+' ex='+!!ex()+' exrev='+!!ex()?.revision+' exload='+!!ex()?.loadingShapes+' red='+!!red()+' msg='+(tools().querySelector('.artifact-process-status')?.textContent||'')+' pressed='+Array.from(tools().querySelectorAll('[data-example-role]')).map(b=>b.getAttribute('aria-pressed')).join());});};
 const at=(x,y)=>s.viewer.viewport.pixelFromPoint(s.layers.find(l=>l.type==='image').item.imageToViewportCoordinates(x,y));
 const tool=name=>document.querySelector('[data-tool='+name+']').click();
 const drag=async(points,event={})=>{
  s.viewer.raiseEvent('canvas-press',{position:at(...points[0]),originalEvent:event});
  for(const point of points.slice(1))s.viewer.raiseEvent('canvas-drag',{position:at(...point),originalEvent:event});
  s.viewer.raiseEvent('canvas-release',{position:at(...points.at(-1)),originalEvent:event});
  await delay(20);
  await until(()=>settled()&&!s.brushBusy);
 };
 const key=k=>document.dispatchEvent(new KeyboardEvent('keydown',{key:k,ctrlKey:k==='z',bubbles:true}));
 tools().querySelector('[data-example-role=negative]').click();
 // The message comes first; the toolbar settles a moment later on a busy machine.
 await until(()=>tools().textContent.includes('Drawing paused.')&&drawTarget()===null&&!document.querySelector('#drawtools').hidden&&document.querySelector('#drawtools [data-tool=polygon]').disabled);
 assert(true,'A failed class switch pauses drawing with the toolbar visible instead of adding to the blood-clot layer');
 assert(Array.from(tools().querySelectorAll('[data-example-role]')).every(b=>b.getAttribute('aria-pressed')==='false'),'Failed activation never claims that a drawing class is selected');
 window.fetch=async(url,options)=>{if(String(url).endsWith('/open_examples'))await delay(150);return originalFetch(url,options);};
 tools().querySelector('[data-example-role=positive]').click();tools().querySelector('[data-example-role=negative]').click();
 await until(()=>drawTarget()===ex()&&ex()?.default_class==='Good tissue');
 assert(tools().querySelector('[data-example-role=negative]').getAttribute('aria-pressed')==='true','Rapid red-to-green activation keeps the last chosen class');
 window.fetch=originalFetch;
 await mark('positive');tool('rect');await drag([[45,55],[150,200]]);
 const markButton=role=>tools().querySelector('[data-example-role='+role+']');
 assert(markButton('positive').getAttribute('aria-pressed')==='true'&&getComputedStyle(markButton('positive')).backgroundColor!==getComputedStyle(markButton('negative')).backgroundColor,'The selected Artifact button has a visible pressed highlight');
 assert(red().features.length===1&&red().features[0].properties.source==='manual_artifact','Rectangle toolbar draws a saved red artifact example');
 assert(getComputedStyle(document.querySelector('[data-layer="'+red().id+'"] path')).stroke!=='rgb(255, 255, 255)','Red remains visible even while an example is selected');
 await mark('negative');
 assert(s.tool==='rect'&&markButton('negative').getAttribute('aria-pressed')==='true'&&markButton('positive').getAttribute('aria-pressed')==='false','Good tissue switches the highlighted class and preserves the drawing tool');
 tool('polygon');
 for(const point of [[260,55],[335,55],[335,200],[260,200]])s.viewer.raiseEvent('canvas-click',{quick:true,position:at(...point),originalEvent:{}});
 key('Enter');await until(()=>ex().features.length===1&&settled());
 assert(ex().features[0].properties.example_role==='negative'&&shapeColor(ex().features[0])==='#2ecc71','Polygon toolbar draws a separate green good-tissue example');
 await mark('positive');tool('freehand');await drag([[45,230],[150,230],[150,315],[45,315],[45,230]]);
 assert(red().features.length===2,'Freehand adds to the same red artifact list');
 tool('rect');document.querySelector('[data-mode=select]').click();await drag([[40,45],[160,325]]);
 assert(s.picks.layer===red()&&s.picks.ids.size===2,'Selection drawing chooses the two red examples, not underlying Tissue');
 key('Delete');await until(()=>settled());
 assert(red().features.length===0&&ex().features.length===1,'Delete removes multiple selected examples and keeps the green one');
 key('z');await until(()=>red().features.length===2&&settled());
 assert(true,'Ctrl+Z restores the red examples');
 const rightClick=(x,y)=>{const point=at(x,y),bounds=s.viewer.element.getBoundingClientRect();s.viewer.canvas.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,button:2,clientX:bounds.left+point.x,clientY:bounds.top+point.y}));};
 tool('off');s.viewer.raiseEvent('canvas-click',{quick:true,position:at(80,90),originalEvent:{}});
 rightClick(80,90);button(document.querySelector('#ctxmenu'),'Delete').click();await until(()=>red().features.length===1&&settled());
 assert(!document.querySelector('#ctxmenu'),'Right-click deletes the selected red example in Move without a competing context menu');
 key('z');await until(()=>red().features.length===2&&settled());
 assert(true,'Undo restores a right-click-deleted example');
 tool('off');s.viewer.raiseEvent('canvas-click',{quick:true,position:at(80,90),originalEvent:{}});
 await until(()=>s.selected?.layer===red());s.brushRadius=5;
 const scroll={position:at(80,90),scroll:1,originalEvent:{altKey:true}};s.viewer.raiseEvent('canvas-scroll',scroll);
 assert(scroll.preventDefaultAction&&s.brushRadius>5,'Alt-wheel changes the example brush size');
 const before=featureArea(red().features.find(f=>f.properties.id===s.selected.id));
 await drag([[80,90],[95,110]],{altKey:true});
 assert(featureArea(red().features.find(f=>f.properties.id===s.selected.id))<before,'Alt erases from a training polygon with the shared brush');
 await drag([[60,70],[25,70]],{shiftKey:true});
 assert(red().features.every(f=>f.properties.source==='manual_artifact')&&ex().features.every(f=>f.properties.example_role==='negative'),'Shift adds area while preserving all red/green labels');
 assert((await read('/shapes/'+tissue.token)).revision===original,'Editing training examples leaves the Tissue file intact');
 const old=ex().features.map(f=>f.properties.id);await loadShapes(ex());
 assert(old.join(',')===ex().features.map(f=>f.properties.id).join(','),'Example polygons and their roles reload from their saved file');
 assert(!button(tools(),'Keep examples'),'Training needs no separate Keep examples step');
 await until(()=>button(tools(),'Train model')&&!button(tools(),'Train model').disabled);button(tools(),'Train model').click();
 await until(async()=>(await read('/studio/browser-check/artifact_review/clot/state')).versions.length===1);
 let library=await read('/studio/browser-check/artifact_review/clot/state');const first=library.versions[0];
 await until(()=>tools().querySelector('[aria-label="Artifact model version"]').value===first.id);
 assert(library.examples.length===1&&/clot_\d{8}T/.test(first.id),'One click captures current drawings and saves a model with a date/time filename');
 assert(tools().querySelector('[aria-label="Artifact model version"]').selectedOptions[0].textContent.includes(String(new Date(first.created).getFullYear())),'The selected model shows its training date and time');
 const previous=library.examples[0].id;
 await mark('negative');tool('rect');await drag([[260,230],[335,310]]);
 await until(()=>button(tools(),'Train model')&&!button(tools(),'Train model').disabled);button(tools(),'Train model').click();
 await until(async()=>(await read('/studio/browser-check/artifact_review/clot/state')).versions.length===2);
 library=await read('/studio/browser-check/artifact_review/clot/state');const second=library.versions[1];
 await until(()=>tools().querySelector('[aria-label="Artifact model version"]').value===second.id);
 assert(second.id!==first.id&&library.examples[0].id!==previous&&first.examples[0]!==second.examples[0],'Retraining snapshots the new polygons into a separate dated model and keeps the earlier snapshot');
 const models=tools().querySelector('[aria-label="Artifact model version"]');models.value=first.id;models.dispatchEvent(new Event('change'));
 await until(()=>!button(tools(),'Predict').disabled);button(tools(),'Predict').click();
 // An earlier prediction can still be shown for a moment: wait for the chosen model's (a wrong model times out).
 await until(()=>s.layers.find(l=>l.id==='artifact-clot')?.preview&&s.layers.find(l=>l.id==='artifact-clot').features.some(f=>f.properties.model_version===first.id)).catch(()=>{
   const c=s.layers.find(l=>l.id==='artifact-clot');
   throw Error('Predict did not use the chosen model: select='+tools().querySelector('[aria-label="Artifact model version"]').value.slice(-12)+' first='+first.id.slice(-12)+' second='+second.id.slice(-12)+' preview='+!!c?.preview+' versions='+JSON.stringify([...new Set((c?.features||[]).map(f=>String(f.properties.model_version).slice(-12)))])+' status='+(tools().querySelector('.artifact-process-status')?.textContent||''));});
 assert(true,'Predict uses the explicitly selected older model for comparison');
 assert((await read('/studio/browser-check/artifact_review/clot/state')).versions.length===2,'Prediction reuses models without training again');
 assert(ex().features.length===2&&red().features.filter(f=>f.properties.source==='manual_artifact').length===2&&(await read('/shapes/'+tissue.token)).revision===original,'Predictions preserve all example drawings and original Tissue');
 assert(drawTarget()===ex()&&ex().default_class==='Good tissue','Completing prediction does not change the chosen example class or drawing target');
 const artifact=s.layers.find(l=>l.id==='artifact-clot'),predicted=artifact.features.length;
 assert(!button(tools().querySelector('.artifact-polygons'),'Save polygons')&&button(document.querySelector('.studio-runbar'),'Save polygons'),'Artifact previews have one Save polygons action in the fixed footer');
 assert(!document.querySelector('[data-layer-id=artifact-clot] [aria-label="Pinker by"]')&&!button(document.querySelector('[data-layer-id=artifact-clot]'),'Preview'),'The blood-clot layer also omits the legacy slider and detector Preview');
 const feature=artifact.features[0],bounds=featureBounds(feature);let inside;
 for(let y=bounds.y0+.5;y<bounds.y1&&!inside;y+=.5)for(let x=bounds.x0+.5;x<bounds.x1;x+=.5)if([[x,y],[x+2,y],[x-2,y],[x,y+2],[x,y-2]].every(([a,b])=>contains(feature,a,b))){inside=[x,y];break;}
 assert(!!inside,'The predicted artifact has an interior point for the deletion check');
 s.selected={layer:artifact,id:feature.properties.id};s.picks={layer:artifact,ids:new Set([feature.properties.id])};tool('off');
 rightClick(...inside);assert(!!button(document.querySelector('#ctxmenu')||document.createElement('div'),'Delete'),'Predicted artifact context menu offers Delete: '+document.querySelector('#ctxmenu')?.textContent);button(document.querySelector('#ctxmenu'),'Delete').click();await until(()=>artifact.features.length===predicted-1);
 assert(!!artifact.preview,'Right-click deletes a selected artifact preview before Save polygons');
 key('z');await until(()=>artifact.features.length===predicted);
 assert(true,'Undo restores the deleted predicted artifact');
 rightClick(...inside);button(document.querySelector('#ctxmenu'),'Mark as good tissue').click();
 await until(()=>artifact.features.length===predicted-1&&ex().features.length===3&&settled());
 assert(ex().features.some(f=>f.properties.source==='corrected_prediction'),'Right-click converts a predicted polygon into a separate green correction');
 key('z');await until(()=>artifact.features.length===predicted&&ex().features.length===2&&settled());
 assert(true,'Undo restores both the red polygon and the previous good-tissue list');

 button(document.querySelector('.studio-runbar'),'Save polygons').click();await until(()=>!artifact.preview&&!artifact.saveBusy);
 assert((await read('/shapes/'+artifact.token)).features.length===predicted,'The single footer Save polygons writes the reviewed prediction');

 // Follow the same model onto another image, without transferring its drawings.
 const firstToken=s.layers.find(l=>l.type==='image').token,firstExamples=JSON.stringify(ex().features);
 tools().querySelector('tr[data-image="second.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.some(l=>l.type==='image'&&l.token!==firstToken)&&tools().querySelector('[aria-label="Artifact model version"]')?.options.length===2);
 const secondTissue=s.layers.find(l=>l.id==='tissue-regions'),secondOriginal=(await read('/shapes/'+secondTissue.token)).revision;
 const secondModels=tools().querySelector('[aria-label="Artifact model version"]');secondModels.value=second.id;secondModels.dispatchEvent(new Event('change'));
 await until(()=>!button(tools(),'Predict').disabled);button(tools(),'Predict').click();
 await until(()=>s.layers.find(l=>l.id==='artifact-clot')?.preview&&s.layers.find(l=>l.id==='artifact-clot').features.length);
 assert(s.layers.find(l=>l.id==='artifact-clot').features.every(f=>f.properties.model_version===second.id),'A model trained on the first image predicts polygons on the second image');
 await mark('positive');
 assert(ex().features.length===0,'The new image has its own examples rather than copies of the first image drawings');
 const oldCount=red().features.length;tool('rect');await drag([[50,60],[155,205]]);
 assert(red().features.length===oldCount+1,'New red drawings append directly to the prediction list');
 await mark('negative');tool('rect');await drag([[265,60],[340,205]]);
 await until(()=>button(tools(),'Train model')&&!button(tools(),'Train model').disabled);button(tools(),'Train model').click();
 await until(async()=>(await read('/studio/browser-check/artifact_review/clot/state')).versions.length===3);
 library=await read('/studio/browser-check/artifact_review/clot/state');const third=library.versions[2];
 await until(()=>tools().querySelector('[aria-label="Artifact model version"]').value===third.id);
 assert(third.images===2&&library.examples.length===2&&library.examples.some(e=>e.image==='markers.ome.tiff')&&library.examples.some(e=>e.image==='second.ome.tiff'),'Retraining on another image retains both images in a new dated model');
 assert(library.versions.some(v=>v.id===first.id)&&library.versions.some(v=>v.id===second.id),'Both earlier models remain available after training with the second image');
 assert((await read('/shapes/'+secondTissue.token)).revision===secondOriginal,'Cross-image prediction and training preserve the second original Tissue');
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.find(l=>l.type==='image')?.token===firstToken&&tools().querySelector('[data-example-role=positive]'));
 await mark('positive');
 assert(JSON.stringify(ex().features)===firstExamples,'Returning to the first image restores its own unchanged red and green examples');
 await until(()=>tools().querySelector('[aria-label="Artifact model version"]')?.options.length===3);
 assert(true,'The new two-image model is also available when returning to the first image');
 // Train on selected: both reviewed images at once, one new model.
 await until(()=>!button(tools(),'Train on selected').disabled);  // the previous training may still be ending
 button(tools(),'Train on selected').click();
 await until(()=>/Tick the reviewed images/.test(tools().textContent));
 for(const name of ['markers.ome.tiff','second.ome.tiff'])tools().querySelector('[aria-label="Pick '+name+'"]').click();
 const base=tools().querySelector('[aria-label="Artifact model version"]').value;
 await until(()=>!button(tools(),'Train on selected').disabled);
 button(tools(),'Train on selected').click();
 await until(async()=>(await read('/studio/browser-check/artifact_review/clot/state')).versions.length===4);
 library=await read('/studio/browser-check/artifact_review/clot/state');
 assert(library.versions[3].images===2&&library.versions[3].parent_version===base,'Train on selected makes one model from both ticked images, on the chosen model');
 let workflow;
 await until(async()=>(workflow=await read('/studio/browser-check/annotation_workflow')).artifact_libraries.clot.models.length===4);  // recorded as the job ends
 assert(Object.keys(workflow.images).length>=2&&workflow.artifact_libraries.clot.models.length===4,'Project workflow JSON links both images and every model version');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=240, setup=setup)
