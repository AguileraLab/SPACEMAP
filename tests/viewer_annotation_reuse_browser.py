"""Follow an edited threshold annotation into Classification and the final Export panel."""
from viewer_threshold_browser import main
from viewer_studio_regions_browser import DRIVER as REGION_DRIVER, setup

EXTRA = r"""
  // Existing shared toolbar, not a separate editor for generated annotations.
  assert(!tools().textContent.includes('null'),'Marker settings contain no stray null placeholder');
  state.picks={layer:regions(),ids:new Set([regions().features[0].properties.id])};state.selected=null;
  setTool('split');
  viewer.raiseEvent('canvas-press',{position:at(0,100)});viewer.raiseEvent('canvas-drag',{position:at(205,100)});viewer.raiseEvent('canvas-release',{position:at(205,100)});
  await until(()=>!state.brushBusy&&regions().features.length===2);
  assert(regions().features.every(f=>f.properties.classification.name==='Normal'),'Split preserves the parent class on both pieces');
  setTool('off');viewer.raiseEvent('canvas-click',{quick:true,position:at(50,50),originalEvent:{}});
  await until(()=>!document.querySelector('#cellcard').hidden&&document.querySelector('[aria-label="Region name"]'));
  const tumorInput=document.querySelector('[aria-label="Region name"]');tumorInput.value='Tumor';button(document.querySelector('#cellcard'),'Apply label').click();
  await until(()=>regions().features.some(f=>f.properties.classification.name==='Tumor')&&!regions().saveBusy&&!regions().savePending);
  const parentFile=regions().saved_file,parentToken=regions().token;
  const beforeParent=await onDisk();
  stage('Classification');await until(()=>tools().querySelector('[aria-label="Work within"]:not(:disabled)'));
  set('Work within','file:'+parentFile);
  await until(()=>tools().querySelector('[aria-label="Annotation name"]:not(:disabled)')?.options.length===3);
  set('Annotation name','Tumor');
  const withinName=()=>{const v=tools().querySelector('[aria-label="Work within"]')?.value||'';return v.startsWith('name:')?JSON.parse(v.slice(5)).name:'';};
  await until(()=>withinName()==='Tumor'&&!tools().querySelector('.annotation-scope [role=status]')?.textContent);
  const barRow=label=>Array.from(document.querySelectorAll('.studio-runbar .row')).find(r=>r.querySelector('.lbl')?.textContent===label);
  const classifyNow=()=>button(barRow('Classify:'),'This image');
  const ready=()=>tools().querySelectorAll('.classification-row').length>0&&!classifyNow()?.disabled;
  await until(()=>classifyNow()&&!classifyNow().disabled);
  set('Number of classes','2');classifyNow().click();await delay(100);await until(()=>ready()&&tools().querySelectorAll('.classification-row').length<=2);
  const within=tools().querySelector('[aria-label="Work within"]').value;
  assert(within.startsWith('name:')&&JSON.parse(within.slice(5)).path===parentFile&&JSON.parse(within.slice(5)).name==='Tumor','Classification reuses the original marker layer without making an import copy (its Tumor polygons)');
  button(barRow('Save polygons:'),'This image').click();
  await until(()=>ready()&&state.layers.some(l=>l.annotation_group==='classes'&&l.name==='Classification · Tumor'&&l.features?.length&&l.editable&&l.visible&&l.saved_file));
  const child=state.layers.find(l=>l.name==='Classification · Tumor');
  assert(child.editable&&child.visible&&child.saved_file!==parentFile,'Subclassifying Tumor saves a separate editable annotation layer');
  assert(withinName()==='Tumor','The saved subclassification retains the selected Tumor scope');
  const currentParent=await (await fetch('/spacemap-viewer/api/shapes/'+parentToken)).json();
  assert(JSON.stringify(currentParent.features)===JSON.stringify(beforeParent),'Subclassifying leaves the edited Tumor and Normal polygons unchanged');
  assert(child.features.every(f=>(f.geometry.type==='Polygon'?f.geometry.coordinates.flat():f.geometry.coordinates.flat(2)).every(p=>p[1]<=101)),'Child polygons stay inside the chosen Tumor piece');
  stage('Export');await until(()=>tools().querySelector('.annotation-rule[data-rule="local:Tumor"]')&&tools().querySelector('.annotation-rule[data-rule="local:Normal"]'));
  assert(true,'Export lists the names, from Marker regions and Classification alike');
  // Tick every annotation for Export (one rule each, for every image).
  const unticked=()=>Array.from(tools().querySelectorAll('.annotation-rule:not(.head) input[type=checkbox][aria-label^="Export "]')).find(box=>!box.checked);
  for(let box=unticked();box;box=unticked()){const rule=box.closest('.annotation-rule').dataset.rule;box.click();
    await until(()=>tools().querySelector('.annotation-rule[data-rule="'+CSS.escape(rule)+'"] input[type=checkbox][aria-label^="Export "]')?.checked);}
  await until(async()=>{const rules=(await (await fetch('/spacemap-viewer/api/studio/browser-check/annotation_export')).json()).settings.rules;return rules['local:Tumor']?.export&&rules['local:Normal']?.export;});
  let exported=null;const originalFetch=window.fetch;
  window.fetch=async(url,options)=>{const response=await originalFetch(url,options);if(String(url).endsWith('/annotation_export')&&options?.method==='POST')exported=await response.clone().json();return response;};
  // Prevent a browser save dialog; inspect the exact returned GeoJSON instead.
  const click=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(!this.download)click.call(this);};
  button(document.querySelector('.studio-runbar'),'This image').click();
  await until(()=>exported&&tools().textContent.includes('annotations exported'));
  assert(exported.layers>=2&&exported.geojson.features.some(f=>f.properties.classification.name==='Normal')&&exported.geojson.features.some(f=>f.properties.classification_run),'Export keeps the edited parent names and child classifications together');
  assert(exported.geojson.features.every(f=>f.properties.source_layer&&f.properties.source_layer_revision),'Export preserves each polygon’s source layer and revision');
  window.fetch=originalFetch;HTMLAnchorElement.prototype.click=click;
  stage('Marker regions');await until(()=>tools().querySelector('[aria-label="Edit marker region layer"]'));
  set('Work within','image');await until(()=>tools().querySelector('[aria-label="Work within"]')?.value==='image'&&!tools().querySelector('.annotation-scope [role=status]')?.textContent);
  set('Name the regions','Epithelia');
  button(row('Detect regions'),'This image').click();
  await until(()=>state.layers.some(l=>l.name==='Marker regions · Epithelia'&&l.features?.length)&&tools().querySelector('[aria-label="Edit marker region layer"]')?.selectedOptions[0]?.textContent==='Marker regions · Epithelia');
  assert(state.layers.find(l=>l.name==='Marker regions · Epithelia').saved_file!==parentFile,'A new region label creates a separate layer and selects it for editing');
  assert((await onDisk()).length===2,'A new result label does not overwrite the previously split and named regions');
  const childFile=child.saved_file;
  tools().querySelector('tr[data-image="roi.ome.tiff"] button.studio-item-name').click();
  await until(()=>state.imagesReady&&state.layers.find(l=>l.type==='image')?.name==='roi.ome.tiff'&&!state.commandBusy);
  tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
  await until(()=>state.imagesReady&&!state.commandBusy&&state.layers.some(l=>l.saved_file===childFile));
  assert(state.layers.find(l=>l.saved_file===childFile).editable&&state.layers.some(l=>l.name==='Marker regions · Epithelia'),'Reopening the image restores editable parent and child annotation layers');
"""
DRIVER=REGION_DRIVER.replace("  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));",EXTRA+"\n  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));")
DRIVER=DRIVER.replace("window.prompt=()=>'Normal';","window.prompt=()=>'Normal';window.confirm=()=>true;")
if __name__=='__main__':main(DRIVER,studio=True,timeout=300,setup=setup)
