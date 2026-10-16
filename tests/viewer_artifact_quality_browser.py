"""One quality tab: old Blur model, drawings, all sliders and a shared model."""
from viewer_threshold_browser import main
from viewer_artifact_training_browser import setup as images


def setup(root):
    from spacemap_gui.studio import artifact_learning as learning
    from test_artifact_learning import polygon
    images(root)
    learning.snapshot(root, 'blur', root / 'markers.ome.tiff', 2,
        [polygon(45, 55, 150, 200)], [polygon(260, 55, 335, 200)],
        [polygon(0, 0, 384, 384)], False)
    learning.train(learning.training_request(root, 'blur'), lambda *a: None)

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
 const library=part=>read('/studio/browser-check/artifact_review/'+part+'/state');
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.some(l=>l.id==='tissue-regions'&&l.revision));
 const tissue=s.layers.find(l=>l.id==='tissue-regions'),original=(await read('/shapes/'+tissue.token)).revision;
 button(document.querySelector('.studio-stage-tabs'),'Artifacts').click();
 await until(()=>tools().querySelector('[data-example-role=positive]'));
 assert(Array.from(tools().querySelectorAll('[aria-label="Artifact type"] button')).map(b=>b.textContent).join('|')==='Blood clots|Image quality','Two artifact sections: Blood clots and shared Image quality');
 const versions={};
 for(const [part,title] of [['debris','Image quality']]){
  button(tools().querySelector('[aria-label="Artifact type"]'),title).click();
  await until(()=>tools().querySelector('[data-artifact-training='+part+']'));
  const red=()=>s.layers.find(l=>l.id==='artifact-'+part),green=()=>s.layers.find(l=>l.id==='artifact-examples-'+part);
  const settled=()=>[red(),green()].every(l=>!l?.savePending&&!l?.saveBusy&&!l?.brushBusy);
  assert(tools().querySelector('[aria-label="Artifact scale"]').selectedOptions[0].textContent==='1/2 · 1.02 µm/px',title+' uses physical-unit scales and a finer default');
  assert(!tools().querySelector('[aria-label="Remaining area is good"]').checked,title+' leaves unreviewed tissue out of training by default');
  {
   assert(tools().querySelector('[aria-label="Blur level"]')&&tools().querySelector('[aria-label="Bright level"]')&&tools().querySelector('[aria-label="Lint signal"]'),'Shared section keeps all three classical sliders');
   const fold=tools().querySelector('[data-artifact-sliders=debris]');assert(!fold.open,'Optional sliders start folded');fold.open=true;
   const models=tools().querySelector('[aria-label="Artifact model version"]');
   assert(models.options.length===1&&models.value.startsWith('blur:')&&models.selectedOptions[0].textContent.startsWith('Blur ·'),'Earlier Blur model is available with its original label');
   versions.blur=models.value;
   button(tools(),'Predict').click();
   await until(()=>red()?.preview&&!button(tools(),'Predict').disabled);
   assert(red().features.some(f=>f.properties.model_part==='blur'),'Earlier Blur model predicts into the shared editable artifact list');
   const detector=tools().querySelector('[aria-label="Preview detector"]');detector.value='lint';detector.dispatchEvent(new Event('change'));
   assert(tools().querySelector('[aria-label="Blur level"]').disabled&&tools().querySelector('[aria-label="Bright level"]').disabled&&!tools().querySelector('[aria-label="Lint signal"]').disabled,'Choose Lint without running Blur or Bright spots');
  }
  await until(()=>red()?.preview&&!button(tools(),'Preview').disabled);
  assert(true,title+' slider creates an editable preview without changing Tissue');
  const at=(x,y)=>s.viewer.viewport.pixelFromPoint(s.layers.find(l=>l.type==='image').item.imageToViewportCoordinates(x,y));
  const draw=async(role,points)=>{
   tools().querySelector('[data-example-role='+role+']').click();
   await until(()=>green()?.revision&&drawTarget()===(role==='positive'?red():green()));
   assert(tools().querySelector('[data-example-role='+role+']').getAttribute('aria-pressed')==='true',title+' highlights '+role+' drawing mode');
   document.querySelector('[data-tool=rect]').click();
   s.viewer.raiseEvent('canvas-press',{position:at(...points[0]),originalEvent:{}});
   s.viewer.raiseEvent('canvas-drag',{position:at(...points[1]),originalEvent:{}});
   s.viewer.raiseEvent('canvas-release',{position:at(...points[1]),originalEvent:{}});
   await delay(50);await until(()=>settled());
  };
  await draw('positive',[[45,55],[150,200]]);
  await draw('negative',[[260,55],[335,200]]);
  assert(red().features.some(f=>f.properties.source==='manual_artifact')&&green().features.length===1,title+' keeps manual red additions in the artifact list and green examples separate');
  assert(shapeColor(red().features.at(-1))==='#e6194b'&&shapeColor(green().features[0])==='#2ecc71',title+' uses consistent red and green polygons');
  await until(()=>!button(tools(),'Train model').disabled);button(tools(),'Train model').click();
  await until(async()=>(await library(part)).versions.length===2);
  await until(()=>button(tools(),'Train model')&&!button(tools(),'Train model').disabled);
  const model=(await library(part)).versions.find(v=>!v.source_part);versions[part]=model.id;
  assert(model.parent_version===versions.blur&&model.artifact_targets.join('|')==='blur|bright|lint','Training creates an Image quality branch from the selected Blur model');
  assert(model.feature_schema==='quality_v2'&&model.mode==='supervised'&&!model.feature_names.some(n=>n.startsWith('marker:'))&&model.feature_names.some(n=>n.startsWith('quality2:z:')),title+' trains a dated quality model across all marker channels (no measurement names a marker)');
  await until(()=>!button(tools(),'Predict').disabled);button(tools(),'Predict').click();
  await until(()=>red().features.some(f=>f.properties.model_version===model.id)&&!button(tools(),'Predict').disabled);
  assert(red().features.some(f=>f.properties.source==='manual_artifact'),title+' prediction retains manual red polygons');
  assert(red().features.filter(f=>f.properties.model_version).every(f=>shapeColor(f)==='#e6194b'),title+' predictions are red and editable');
  // Convert a prediction through the real context menu, then undo both lists.
  document.querySelector('[data-tool=off]').click();
  s.viewer.raiseEvent('canvas-click',{quick:true,position:at(60,250),originalEvent:{}});
  const point=at(60,250),bounds=s.viewer.element.getBoundingClientRect();
  s.viewer.canvas.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,button:2,clientX:bounds.left+point.x,clientY:bounds.top+point.y}));
  await until(()=>document.querySelector('#ctxmenu'));
  button(document.querySelector('#ctxmenu'),part==='blur'?'Mark as sharp tissue':'Mark as good tissue').click();
  await until(()=>green().features.length===2&&settled());
  assert(true,title+' converts a false detection to a green training example');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
  await until(()=>green().features.length===1&&settled());
  assert(true,title+' Undo restores the prediction and green list together');
  const saves=Array.from(tools().querySelectorAll('button')).filter(b=>b.textContent==='Save polygons'&&b.getClientRects().length);
  assert(saves.length===1,title+' offers one Save polygons button');saves[0].click();
  await until(()=>!red().preview&&settled());
  assert((await read('/shapes/'+tissue.token)).revision===original,title+' saves reviewed artifacts without modifying source Tissue');
 }
 assert((await library('clot')).versions.length===0,'Quality training does not change the blood-clot library');
 const token=s.layers.find(l=>l.type==='image').token;
 tools().querySelector('tr[data-image="second.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.find(l=>l.type==='image')?.token!==token&&tools().querySelector('[data-artifact-training=debris]'));
 await until(()=>tools().querySelector('[aria-label="Artifact model version"]')?.value===versions.debris&&s.layers.find(l=>l.id==='artifact-debris')?.image_token===s.layers.find(l=>l.type==='image')?.token&&!button(tools(),'Predict').disabled);
 assert(tools().querySelector('[aria-label="Preview detector"]').closest('[data-artifact-image]').dataset.artifactImage===s.layers.find(l=>l.type==='image').token,'New image controls are bound to that image before Predict is enabled');
 assert(tools().querySelector('[aria-label="Preview detector"]').value==='lint','Combined detector settings persist on another image');
 button(tools(),'Predict').click();
 await until(()=>s.layers.find(l=>l.id==='artifact-debris')?.features.some(f=>f.properties.model_version===versions.debris));
 assert(true,'The shared quality model predicts on a second image');
 const record=await read('/studio/browser-check/annotation_workflow');
 assert(record.artifact_libraries.blur.models.length===1&&record.artifact_libraries.debris.models.length===1,'Workflow JSON preserves the original Blur model and records the new shared model');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=300, setup=setup)
