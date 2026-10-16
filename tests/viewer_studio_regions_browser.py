"""Browser-check Studio's marker regions and the region tools: find, preview, detect, click, name, select, undo.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_regions_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<2400;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message.slice(0,90)));};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent('page error: '+event.message+' '+event.filename+':'+event.lineno)));
window.prompt=()=>'Normal';
setTimeout(async()=>{
 try{
  const state=window.__state;
  const tools=()=>document.querySelector('#studio-tools');
  const stage=name=>Array.from(document.querySelectorAll('.studio-stage-tabs button')).find(item=>item.textContent===name).click();
  const row=text=>Array.from(document.querySelectorAll('.studio-runbar .row')).find(item=>item.querySelector('.lbl')?.textContent===text);
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=value;input.dispatchEvent(new Event('change'));};
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab4]'));
  document.querySelector('[data-step=tab4]').click();
  await until(()=>document.querySelectorAll('.studio-stage-tabs button').length);
  await until(()=>tools().querySelector('#studio-tools tr[data-image="markers.ome.tiff"] button.studio-item-name'));
  tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
  const regions=()=>state.layers.find(item=>item.id==='marker-regions');
  await until(()=>state.imagesReady && state.layers.some(item=>item.type==='image' && item.info?.channels?.includes('PanCK')) && !state.commandBusy);
  stage('Marker regions');
  await until(()=>regions() && tools().querySelector('[aria-label="Marker 1"]')===null && Array.from(tools().querySelectorAll('select')).length && /Region|review its regions|No regions/.test(tools().textContent) && button(tools(),'Add a marker') && !button(tools(),'Add a marker').disabled);
  assert(regions().regions && state.layers.some(item=>item.id==='drawn-artifacts'),'The image opens with its regions layer and its drawn artifacts');
  button(tools(),'Add a marker').click();
  button(tools(),'Add a marker').click();
  set('Marker 1','PanCK');set('Level of marker 1','100');set('Marker 2','ECad');set('Level of marker 2','200');
  set('Work within','image');set('Smoothing sigma','0');set('Region scale','1');set('Smallest region in square micrometres','10');
  await until(()=>!tools().querySelector('.annotation-scope [role=status]')?.textContent);
  assert(tools().textContent.includes('Not saved'),'Changed settings show as not saved (they save on their own)');
  button(tools(),'Preview').click();
  await until(()=>/1 region · /.test(tools().textContent));
  assert(state.previews['marker-regions-preview'],'Preview draws the regions without saving them');
  set('Cutoff','groups');
  await until(()=>tools().querySelector('[aria-label="Fit two groups per"]') && tools().querySelector('[aria-label="Level of marker 1"]').value==='0.9');
  button(tools(),'Preview').click();
  await until(()=>/% sure, 1 ROI, separation/.test(tools().textContent));
  assert(true,'Two groups: a how-sure level per marker, fitted per slide or image, its separation shown after Preview');
  set('Cutoff','raw');set('Level of marker 1','100');set('Level of marker 2','200');
  button(row('Detect regions'),'This image').click();
  await until(()=>(regions().features||[]).length===1 && tools().querySelector('.region-list .shape-row'));
  assert(/1 marker region in markers.ome.tiff/.test(document.querySelector('.studio-runbar').textContent),'Detect saves the region for the open image');
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/marker_regions')).json();
  assert(saved.values.rules.length===2 && saved.values.rules[1].above===200,'The settings are saved with the project');
  // Click it: its card with the markers high in it.
  const image=state.layers.find(item=>item.type==='image'), viewer=state.viewer;
  const at=(x,y)=>viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(x,y));
  viewer.raiseEvent('canvas-click',{quick:true,position:at(50,50),originalEvent:{}});
  await until(()=>!document.querySelector('#cellcard').hidden && /High here/.test(document.querySelector('#cellcard').textContent));
  const card=document.querySelector('#cellcard');
  assert(/^(ECad|PanCK)/.test(card.querySelector('.region-marker').textContent) && document.querySelector('#shapes path.shape.selected'),'Clicking a region outlines it and its card ranks its markers');
  const name=card.querySelector('[aria-label="Region name"]');name.value='Tumor';button(card,'Apply label').click();
  await until(()=>regions().features[0].properties.name==='Tumor' && /Saved/.test(regions().status||''));
  const token=regions().token;
  const onDisk=async()=>(await (await fetch('/spacemap-viewer/api/shapes/'+token)).json()).features;
  await until(async()=>(await onDisk())[0]?.properties.name==='Tumor');
  assert(true,'Naming from the card saves the name into the regions file');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
  await until(()=>regions().features[0].properties.name==='Region 1');
  assert(true,'Ctrl+Z takes the name back');
  // Right click: name from the list or a new name.
  // Selected annotations now use the shared right-click deletion gesture.
  // Click outside first to request the naming menu for an unselected region.
  viewer.raiseEvent('canvas-click',{quick:true,position:at(350,350),originalEvent:{}});
  const canvas=document.querySelector('#osd .openseadragon-canvas');const box=canvas.getBoundingClientRect();
  const p=at(50,50);
  canvas.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:box.left+p.x,clientY:box.top+p.y}));
  await until(()=>document.querySelector('#ctxmenu'));
  assert(/Name Region 1 as/.test(document.querySelector('#ctxmenu').textContent),'Right-clicking a region offers to name it');
  button(document.querySelector('#ctxmenu'),'New name…').click();
  await until(()=>regions().features[0].properties.name==='Normal' && regions().features[0].properties.named);
  assert(true,'A new name given from the right-click menu names the region');
  // Select: an outline selects the regions inside it; Delete removes them; Ctrl+Z brings them back.
  document.querySelector('#drawtools [data-mode="select"]').click();
  assert(state.selectMode && state.tool==='rect','Select switches the drawing tools to selecting');
  viewer.raiseEvent('canvas-press',{position:at(5,5)});viewer.raiseEvent('canvas-drag',{position:at(390,390)});viewer.raiseEvent('canvas-release',{position:at(390,390)});
  await until(()=>state.picks && state.picks.ids.size===1);
  assert(regions().features.length===1,'A selecting outline draws nothing');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true}));
  await until(()=>regions().features.length===0);
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
  await until(()=>regions().features.length===1);
  assert(true,'Delete removes the selected regions and Ctrl+Z brings them back');
  document.querySelector('#drawtools [data-mode="select"]').click();
  // The list: sort by size, remove the small ones.
  await until(()=>tools().querySelector('[aria-label="Remove regions smaller than (µm²)"]'));
  const smallest=tools().querySelector('[aria-label="Remove regions smaller than (µm²)"]');smallest.value='1000000';
  button(tools(),'Remove smaller').click();
  await until(()=>regions().features.length===0);
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
  await until(()=>regions().features.length===1);
  assert(true,'Remove smaller drops regions below a size; Ctrl+Z brings them back');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+(document.querySelector('#studio-tools')?.textContent||'').slice(0,1500)));}
},100);
</script>
"""


def setup(root):
    """An image with PanCK in its left half, ECad in its top half and CD3 in a small square, next to the check's images."""
    import numpy as np
    import tifffile

    planes = np.full((3, 400, 400), 20, np.uint16)
    planes[0, :, :200] = 500
    planes[1, :200, :] = 800
    planes[2, 300:320, 300:320] = 900
    tifffile.imwrite(root / "markers.ome.tiff", planes, metadata={"axes": "CYX", "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0,
                                                                  "Channel": {"Name": ["PanCK", "ECad", "CD3"]}})


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300, setup=setup)
