"""Real browser: bulk labels survive refresh; tissue minus named regions creates a reusable layer."""
from viewer_threshold_browser import main
from viewer_studio_regions_browser import DRIVER as REGION_DRIVER, setup as marker_setup


def setup(root):
    from shapely.geometry import box, mapping
    from spacemap_gui.studio import annotation, annotation_edits as edits
    marker_setup(root)
    output = annotation.all_folders(root, {})['output']
    path = annotation.valid_tissue(output, 'markers')
    edits.replace(path, [{'type': 'Feature', 'geometry': mapping(box(0, 0, 400, 400)),
                          'properties': {'name': 'tissue', 'classification': {'name': 'Valid_Tissue'}}}],
                  edits.revision(path), 'Fixture')


EXTRA = r"""
  await until(()=>!regions().saveBusy&&!regions().savePending);
  state.picks={layer:regions(),ids:new Set([regions().features[0].properties.id])};state.selected=null;
  setTool('split');
  viewer.raiseEvent('canvas-press',{position:at(0,100)});viewer.raiseEvent('canvas-drag',{position:at(205,100)});viewer.raiseEvent('canvas-release',{position:at(205,100)});
  await until(()=>!state.brushBusy&&regions().features.length===2);
  setTool('off');
  const list=()=>tools().querySelector('#studio-marker-regions');
  const ticks=()=>Array.from(list().querySelectorAll('.shape-row input[type=checkbox]'));
  for(const id of ticks().filter(t=>t.checked).map(t=>t.dataset.regionId))list().querySelector('[data-region-id="'+id+'"]').click();
  let draft=list().querySelector('[aria-label="Name the selected regions"]');
  draft.value='Epithelia';draft.dispatchEvent(new Event('input'));draft.focus();draft.setSelectionRange(3,3);
  assert(button(list(),'Apply label').disabled,'Apply label requires selected polygons');
  buildPanel();refreshRegionList();
  draft=list().querySelector('[aria-label="Name the selected regions"]');
  assert(draft.value==='Epithelia'&&document.activeElement===draft&&draft.selectionStart===3,'A list refresh preserves the label draft, focus and caret');
  ticks()[0].click();ticks()[1].click();
  assert(list().querySelector('[aria-label="Name the selected regions"]').value==='Epithelia'&&!button(list(),'Apply label').disabled,'Selecting multiple polygons keeps the typed label and enables Apply label');
  button(list(),'Apply label').click();
  await until(()=>!regions().saveBusy&&!regions().savePending&&regions().features.every(f=>f.properties.name==='Epithelia'));
  assert(Array.from(list().querySelectorAll('.region-name')).every(el=>el.textContent==='Epithelia'),'Bulk labels appear beside every selected polygon');
  assert((await onDisk()).every(f=>f.properties.classification.name==='Epithelia'),'Bulk labels are saved in GeoJSON for later reuse');
  draft=list().querySelector('[aria-label="Name the selected regions"]');draft.blur();
  button(list(),'Undo').click();
  await until(()=>regions().features.every(f=>f.properties.name==='Normal')&&!regions().saveBusy&&!regions().savePending);
  assert(true,'Undo restores the previous polygon labels');
  // Give the two pieces distinct labels using the same list controls and Enter.
  ticks()[1].click();
  draft=list().querySelector('[aria-label="Name the selected regions"]');draft.value='Tumor';draft.dispatchEvent(new Event('input'));
  draft.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  await until(()=>regions().features.some(f=>f.properties.name==='Tumor')&&!regions().saveBusy&&!regions().savePending);
  ticks()[0].click();ticks()[1].click();
  draft=list().querySelector('[aria-label="Name the selected regions"]');draft.value='Epithelia';draft.dispatchEvent(new Event('input'));button(list(),'Apply label').click();
  await until(()=>regions().features.some(f=>f.properties.name==='Epithelia')&&!regions().saveBusy&&!regions().savePending);
  assert(new Set((await onDisk()).map(f=>f.properties.name)).size===2,'Different selected pieces retain their own labels');
  const parentBefore=JSON.stringify(await onDisk());
  const diff=()=>tools().querySelector('.annotation-difference');diff().open=true;
  await until(()=>diff().querySelector('[aria-label="From annotations"]')?.options.length>1&&!button(diff(),'Refresh choices').disabled);
  const names=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/annotation_difference')).json()).names;
  const keyOf=async(file,name=null)=>(await names()).find(n=>Object.keys(n.revisions).some(p=>p.endsWith(file))&&(!name||n.name===name)).key;
  const tissueKey=await keyOf('markers_valid_tissue.geojson');
  const regionsFile=regions().saved_file.split('/').pop();
  const choose=async(side,file,name=null)=>{const select=diff().querySelector('[aria-label="'+side+'"]');select.value=await keyOf(file,name);select.dispatchEvent(new Event('change'));};
  assert(Array.from(diff().querySelectorAll('[aria-label="From annotations"] optgroup')).map(g=>g.label)[0]==='markers' &&
    !/All polygons/.test(diff().querySelector('[aria-label="From annotations"]').textContent),'The lists are the names of Layers, grouped by image and file');
  const resultLabel=diff().querySelector('[aria-label="Subtraction result label"]');resultLabel.value='Stroma';resultLabel.dispatchEvent(new Event('input'));
  await choose('From annotations','markers_valid_tissue.geojson');await choose('Subtract annotations',regionsFile,'Tumor');await choose('Subtract annotations',regionsFile,'Epithelia');
  assert(diff().querySelectorAll('.annotation-operand').length===3&&!button(diff(),'Preview').disabled,'Equation accepts one base and multiple named exclusions');
  button(diff(),'Preview').click();
  await until(()=>diff().textContent.includes('remaining polygon')&&!button(diff(),'Create layer').disabled);
  assert(state.previews['annotation-difference-preview']&&!state.layers.some(l=>l.name==='Stroma'),'Subtraction preview shows the remainder without saving a layer');
  // A label edited while the equation is open must invalidate its old preview and choices.
  const tumorFeature=regions().features.find(f=>f.properties.name==='Tumor');
  state.picks={layer:regions(),ids:new Set([tumorFeature.properties.id])};refreshRegionList();
  draft=list().querySelector('[aria-label="Name the selected regions"]');draft.value='Cancer';draft.dispatchEvent(new Event('input'));button(list(),'Apply label').click();
  await until(()=>diff().textContent.includes('Choices updated after annotation edits')&&!regions().saveBusy&&!regions().savePending&&!button(diff(),'Refresh choices').disabled);
  assert(!state.previews['annotation-difference-preview']&&button(diff(),'Create layer').disabled,'Editing a source label invalidates the preview and prevents use of a missing class');
  const missing=Array.from(diff().querySelectorAll('.annotation-operand')).find(chip=>chip.textContent.startsWith('Tumor'));
  missing.querySelector('button').click();await choose('Subtract annotations',regionsFile,'Cancer');
  const updatedParentBefore=JSON.stringify(await onDisk());
  button(diff(),'Preview').click();await until(()=>diff().textContent.includes('remaining polygon')&&!button(diff(),'Create layer').disabled);
  button(diff(),'Create layer').click();
  await until(()=>state.layers.some(l=>l.name==='Stroma'&&l.features?.length)&&diff().textContent.includes('saved as Stroma'));
  const stroma=state.layers.find(l=>l.name==='Stroma');
  assert(stroma.editable&&stroma.visible&&list().querySelector('.region-name').textContent==='Stroma','The remainder is an editable layer, selected in the polygon list with its label');
  assert(stroma.features.every(f=>f.properties.name==='Stroma')&&JSON.stringify(await onDisk())===updatedParentBefore,'Subtraction labels its result and preserves its source polygons');
  const area=stroma.features.reduce((sum,f)=>sum+featureArea(f),0);
  assert(area>115000&&area<125000,'The saved remainder has the expected tissue-minus-regions area');
  button(list(),'Undo').click();await until(()=>!stroma.features.length&&!stroma.saveBusy&&!stroma.savePending);
  assert(true,'Undo reverses creation without deleting source annotations');
  document.activeElement?.blur();document.dispatchEvent(new KeyboardEvent('keydown',{key:'y',ctrlKey:true,bubbles:true}));
  await until(()=>stroma.features.length&&!stroma.saveBusy&&!stroma.savePending);
  const stromaFile=stroma.saved_file;
  // Selected: the same names on each ticked image, as a job.
  await until(()=>!button(diff(),'Selected').disabled).catch(()=>{throw Error('Selected stays disabled: '+diff().textContent.slice(0,300));});
  button(diff(),'Selected').click();
  await until(()=>/Tick the images in the list first/.test(diff().textContent));
  tools().querySelector('[aria-label="Pick markers.ome.tiff"]').click();
  resultLabel.value='Stroma batch';resultLabel.dispatchEvent(new Event('input'));
  button(diff(),'Selected').click();
  await until(async()=>((await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs||[]).some(j=>j.kind==='subtract_annotations'&&j.status==='done'&&j.result.count===1));
  assert(true,'Selected makes the layer on each ticked image as a job');
  tools().querySelector('[aria-label="Pick markers.ome.tiff"]').click();
  resultLabel.value='Stroma';resultLabel.dispatchEvent(new Event('input'));
  stage('Classification');await until(()=>tools().querySelector('[aria-label="Work within"]:not(:disabled)'));
  set('Work within','file:'+stromaFile);
  await until(()=>tools().querySelector('[aria-label="Annotation name"]:not(:disabled)')?.options.length===2);
  assert(tools().querySelector('[aria-label="Work within"]').value==='file:'+stromaFile,'The subtraction result can be selected for Classification');
  stage('Export');await until(()=>tools().querySelector('.annotation-rule[data-rule="local:Stroma"]'));
  assert(true,'The remainder is available in Export');
  tools().querySelector('tr[data-image="roi.ome.tiff"] button.studio-item-name').click();
  await until(()=>state.imagesReady&&!state.commandBusy&&state.layers.find(l=>l.type==='image')?.name==='roi.ome.tiff');
  tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
  await until(()=>state.imagesReady&&!state.commandBusy&&state.layers.some(l=>l.saved_file===stromaFile));
  assert(state.layers.find(l=>l.saved_file===stromaFile).editable,'Reopening the image restores the editable subtraction layer');
  // Delete the result in Layers, then make it again: it comes back.
  await until(()=>document.querySelector('[data-annotation-name="Stroma"] button[aria-label="Delete Stroma"]'));
  document.querySelector('[data-annotation-name="Stroma"] button[aria-label="Delete Stroma"]').click();
  await until(()=>!state.layers.some(l=>l.saved_file===stromaFile));
  stage('Marker regions');
  await until(()=>diff());diff().open=true;
  await until(()=>diff().querySelector('[aria-label="From annotations"]')?.options.length>1&&!button(diff(),'Refresh choices').disabled);
  assert(!(await names()).some(n=>n.name==='Stroma'),'The deleted result is no longer offered');
  const again=diff().querySelector('[aria-label="Subtraction result label"]');again.value='Stroma';again.dispatchEvent(new Event('input'));
  if(!diff().querySelectorAll('.annotation-operand').length){
    await choose('From annotations','markers_valid_tissue.geojson');await choose('Subtract annotations',regionsFile,'Cancer');await choose('Subtract annotations',regionsFile,'Epithelia');
  }
  await until(()=>!button(diff(),'Create layer').disabled);
  button(diff(),'Create layer').click();
  await until(()=>state.layers.some(l=>l.name==='Stroma'&&l.visible&&l.features?.length)&&diff().textContent.includes('saved as Stroma'));
  assert(document.querySelector('[data-annotation-name="Stroma"]'),'A deleted subtraction layer comes back when it is made again');
"""

DRIVER = REGION_DRIVER.replace("  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));",
                               EXTRA + "\n  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));")
DRIVER = DRIVER.replace("window.prompt=()=>'Normal';", "window.prompt=()=>'Normal';window.confirm=()=>true;")
if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=300, setup=setup)
