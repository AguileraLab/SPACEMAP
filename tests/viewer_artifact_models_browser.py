"""All artifact tabs: recover panel loads, choose/delete/restore/branch, bulk flip."""
from viewer_threshold_browser import main
from viewer_artifact_training_browser import setup as images


def setup(root):
    from test_artifact_model_versions import submission
    from spacemap_gui.studio import artifact_learning as learning
    images(root)
    for part in ('clot', 'blur', 'debris'):
        submission(root, part, 'markers.ome.tiff', 1)
        learning.train(learning.training_request(root, part), lambda *a: None)
        submission(root, part, 'markers.ome.tiff', 2, flipped=True)
        submission(root, part, 'second.ome.tiff', 3)
        learning.train(learning.training_request(root, part), lambda *a: None)


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 const read=p=>fetch(api(p)).then(r=>r.json()),library=part=>read('/studio/browser-check/artifact_review/'+part+'/state');
 const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change'));};
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.some(l=>l.id==='tissue-regions'&&l.revision));
 button(document.querySelector('.studio-stage-tabs'),'Artifacts').click();
 await until(()=>tools().querySelector('[data-artifact-training=clot]'));
 const realFetch=window.fetch;let fail=true;
 window.fetch=(url,options)=>String(url).endsWith('/artifact_review/debris/state')&&fail?Promise.resolve(new Response(JSON.stringify({error:'Choose an artifact type.'}),{status:400,headers:{'Content-Type':'application/json'}})):realFetch(url,options);
 button(tools().querySelector('[aria-label="Artifact type"]'),'Image quality').click();
 await until(()=>button(tools(),'Retry'));
 assert(!button(document.querySelector('.studio-runbar'),'Save polygons'),'Failed artifact panel clears controls for the previous artifact');
 fail=false;button(tools(),'Retry').click();
 await until(()=>tools().querySelector('[data-artifact-training=debris]'));
 assert(!!tools().querySelector('[aria-label="Blur level"]')&&!!tools().querySelector('[aria-label="Bright level"]')&&!!tools().querySelector('[aria-label="Lint signal"]'),'Retry restores all quality detectors and learning controls after an unavailable service');
 window.fetch=realFetch;
 const legacy=(await library('debris')).versions.find(v=>v.source_part==='blur');
 const qualityModels=()=>tools().querySelector('[aria-label="Artifact model version"]');
 change(qualityModels(),legacy.id);
 await until(()=>!button(tools(),'Predict').disabled);
 assert((await library('debris')).selected_version===legacy.id,'The shared model picker selects a retained Blur version');
 button(tools(),'Delete model').click();
 await until(()=>qualityModels().options.length===3&&!button(tools(),'Delete model').disabled);
 assert((await library('blur')).deleted_versions[0].id===legacy.id.slice(5),'Deleting a retained Blur version hides only that model');
 const archivedBlur=tools().querySelector('[data-deleted-models=debris]');archivedBlur.open=true;button(archivedBlur,'Restore').click();
 await until(()=>qualityModels().options.length===4&&qualityModels().value===legacy.id&&!button(tools(),'Predict').disabled);
 assert(!(await library('blur')).deleted_versions.length,'Restore returns the original Blur model to the shared picker');
 const oldModels={};
 for(const [part,title] of [['clot','Blood clots'],['debris','Image quality']]){
  button(tools().querySelector('[aria-label="Artifact type"]'),title).click();
  await until(()=>tools().querySelector('[data-artifact-training='+part+']'));
  const goodPanel=tools().querySelector('.artifact-good-tissue');
  assert(goodPanel.nextElementSibling?.classList.contains('artifact-polygons')&&
    !goodPanel.closest('[data-artifact-training]'),title+' places Good tissue directly above Artifact polygons, outside the model controls');
  const trainingPanel=tools().querySelector('[data-artifact-training='+part+']');trainingPanel.open=false;
  goodPanel.querySelector('details').open=true;
  await until(()=>goodPanel.querySelector('.region-list'));
  assert(goodPanel.querySelector('.region-list').getBoundingClientRect().height>0,title+' keeps Good tissue review usable while training controls are folded');
  trainingPanel.open=true;
  const models=()=>tools().querySelector('[aria-label="Artifact model version"]');
  const initial=await library(part),own=initial.versions.filter(v=>!v.source_part),old=own[0],newer=own[1];oldModels[part]=old.id;
  change(models(),old.id);
  await until(()=>!button(tools(),'Predict').disabled);
  assert((await library(part)).selected_version===old.id,title+' persists selection of the older model');
  button(tools(),'Delete model').click();
  await until(()=>models().options.length===initial.versions.length-1&&!button(tools(),'Delete model').disabled);
  const deleted=await library(part);
  assert(deleted.deleted_versions[0].id===old.id&&deleted.selected_version===newer.id,title+' deletes only the chosen model and selects an available version');
  const archived=tools().querySelector('[data-deleted-models='+part+']');archived.open=true;button(archived,'Restore').click();
  await until(()=>models().options.length===initial.versions.length&&models().value===old.id&&!button(tools(),'Predict').disabled);
  assert(!(await library(part)).deleted_versions.length,title+' restores the dated version and its selection');
  const red=()=>s.layers.find(l=>l.id==='artifact-'+part),green=()=>s.layers.find(l=>l.id==='artifact-examples-'+part);
  const settled=()=>[red(),green()].every(l=>!l?.savePending&&!l?.saveBusy&&!l?.brushBusy)&&!s.brushBusy;
  const mark=async role=>{tools().querySelector('[data-example-role='+role+']').click();await until(()=>green()?.revision&&drawTarget()===(role==='positive'?red():green()));};
  const at=(x,y)=>s.viewer.viewport.pixelFromPoint(s.layers.find(l=>l.type==='image').item.imageToViewportCoordinates(x,y));
  await mark('positive');document.querySelector('[data-tool=rect]').click();
  for(const points of [[[30,40],[100,120]],[[140,40],[210,120]],[[250,40],[320,120]]]){
   s.viewer.raiseEvent('canvas-press',{position:at(...points[0]),originalEvent:{}});
   s.viewer.raiseEvent('canvas-drag',{position:at(...points[1]),originalEvent:{}});
   s.viewer.raiseEvent('canvas-release',{position:at(...points[1]),originalEvent:{}});
   await delay(30);await until(()=>settled());
  }
  assert(red().features.length===3,title+' shares the standard polygon drawing tools');
  const original=red().features.map(f=>featureArea(f));
  const rows=()=>tools().querySelector('.artifact-polygons');
  rows().querySelector('.region-selection input[type=checkbox]').click();
  rows().querySelector('.region-selection input[type=checkbox]').click();
  for(let i=0;i<2;i++)rows().querySelectorAll('[data-region-id]')[i].click();
  const label=part==='blur'?'sharp tissue':'good tissue';
  assert(s.picks.ids.size===2&&!button(rows(),'Mark as '+label).disabled,title+' enables batch conversion for two checked polygons');
  button(rows(),'Mark as '+label).click();
  await until(()=>green().features.length===2&&red().features.length===1&&settled());
  assert(green().features.every((f,i)=>featureArea(f)===original[i]),title+' moves both polygons to green without changing boundaries');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
  await until(()=>green().features.length===0&&red().features.length===3&&settled());
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'y',ctrlKey:true,bubbles:true}));
  await until(()=>green().features.length===2&&red().features.length===1&&settled());
  assert(true,title+' Undo and Redo restore the whole batch in both lists');
  const list=tools().querySelector('.artifact-example-list');list.parentElement.open=true;
  list.querySelector('.region-selection input[type=checkbox]').click();
  document.querySelector('[data-tool=off]').click();
  const point=at(50,65),bounds=s.viewer.element.getBoundingClientRect();
  s.viewer.canvas.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,button:2,clientX:bounds.left+point.x,clientY:bounds.top+point.y}));
  await until(()=>document.querySelector('#ctxmenu'));
  assert(!!button(document.querySelector('#ctxmenu'),'Mark selected as artifact'),title+' right-click applies to the selected green batch');
  button(document.querySelector('#ctxmenu'),'Mark selected as artifact').click();
  await until(()=>green().features.length===0&&red().features.length===3&&settled());
  assert(red().features.every(f=>f.properties.source==='manual_artifact'),title+' converted red polygons remain protected manual training examples');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
  await until(()=>green().features.length===2&&red().features.length===1&&settled());
  assert(true,title+' reverse conversion also supports paired Undo');
 }
 // Retrain the selected historical snapshot alone, ignoring the later library and current drawings.
 const part='debris',before=await library(part),old=before.versions.find(v=>v.id===oldModels[part]);
 const advanced=tools().querySelector('[data-training-options=debris]');advanced.open=true;
 advanced.querySelector('[aria-label="Include current image"]').click();
 await until(()=>!button(tools(),'Train model').disabled);button(tools(),'Train model').click();
 await until(async()=>(await library(part)).versions.length===before.versions.length+1);
 const after=await library(part),child=after.versions.at(-1);
 assert(child.parent_version===old.id&&JSON.stringify(child.examples)===JSON.stringify(old.examples),'Retraining an older version uses its exact archived examples and records parent lineage');
 assert(child.images===1&&after.examples.length===2,'Historical retraining excludes later images in the reviewed library');
 await until(()=>tools().querySelector('[aria-label="Artifact model version"]').value===child.id&&button(tools(),'Train model')&&!button(tools(),'Train model').disabled);
 change(tools().querySelector('[aria-label="Artifact model version"]'),old.id);
 await until(()=>!button(tools(),'Predict').disabled);
 const token=s.layers.find(l=>l.type==='image').token;
 tools().querySelector('tr[data-image="second.ome.tiff"] button.studio-item-name').click();
 await until(()=>{const opened=s.layers.find(l=>l.type==='image');return s.imagesReady&&opened&&opened.token!==token&&tools().querySelector('[data-artifact-training=debris]')?.closest('[data-artifact-image]')?.dataset.artifactImage===opened.token;});
 assert(tools().querySelector('[aria-label="Artifact model version"]').value===old.id,'The chosen older model remains selected on another image');
 const workflow=await read('/studio/browser-check/annotation_workflow');
 assert(workflow.artifact_libraries.debris.model_selection.version===old.id&&workflow.artifact_libraries.debris.models.some(v=>v.parent_version===old.id),'Workflow JSON retains model selection and branch lineage');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,2000)));
}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=300, setup=setup)
