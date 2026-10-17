"""Browser-check Studio's Annotation step (tissue) on a disposable image in headless Firefox.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_annotation_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<2400;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab4]'));
  document.querySelector('[data-step=tab4]').click();
  const stage=name=>{const b=Array.from(document.querySelectorAll('.studio-stage-tabs button')).find(item=>item.textContent===name);b.click();};
  const row=text=>Array.from(document.querySelectorAll('.studio-runbar .row')).find(item=>item.querySelector('.lbl')?.textContent===text);
  await until(()=>document.querySelectorAll('.studio-stage-tabs button').length);
  assert(Array.from(document.querySelectorAll('.studio-stage-tabs button')).map(b=>b.textContent).join()==='Tissue,Artifacts,Classification,Marker regions,Zones,Export','The stages are tabs, in pipeline order (necks and pieces are part of Tissue)');
  stage('Zones');
  await until(()=>/Export the pixel classes first/.test(document.querySelector('.studio-method').textContent));
  assert(true,'Zones wait for the exported domains');
  stage('Tissue');
  await until(()=>document.querySelector('[aria-label="Tissue mode"]'));
  assert(document.querySelector('[aria-label="Tissue label"]').value==='tissue' && document.querySelector('[aria-label="Detection scale"]').options.length===5,'The tissue label and the detection scale (pyramid levels) are there');
  const holes=document.querySelector('[aria-label="Fill holes smaller than (µm²)"]');
  assert(holes.value==='0','Hole filling starts off so existing detection is unchanged');
  holes.value='25';holes.dispatchEvent(new Event('change'));
  await until(()=>document.querySelector('#studio-tools tr[data-image] .studio-stages-done'));
  assert(!button(document.querySelector('#studio-tools'),'Pick all') && document.querySelector('#studio-tools .studio-pairs th input[type=checkbox]'),'The images list has the box on top');
  const viewer=state.viewer;
  const mode=document.querySelector('[aria-label="Tissue mode"]');
  mode.value='codex_only';mode.dispatchEvent(new Event('change'));
  assert(document.querySelector('#studio-tools').textContent.includes('Not saved'),'A changed setting shows as not saved');
  const bar=document.querySelector('.studio-runbar');
  button(document.querySelector('#studio-tools'),'Save').click();
  await until(()=>bar.textContent.includes('Settings saved'));
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/tissue_settings')).json();
  assert(saved.values.tissue_mask_mode==='codex_only','Saved into Tab 4\'s tissue settings');
  assert(saved.values.fill_holes_um2===25,'The hole-size setting is saved with the project');
  await until(()=>!button(row('Detect tissue'),'This image').disabled);
  button(row('Detect tissue'),'This image').click();
  await until(()=>/Saved \d+ tissue polygons|rror/.test(bar.textContent));
  const tissueLayer=state.layers.find(l=>l.id==='tissue-regions');
  assert(/Saved \d+ tissue polygons/.test(bar.textContent) && tissueLayer?.features.length>0 && tissueLayer.visible,'Detect tissue saves the tissue at once and shows it as a layer: '+bar.textContent.slice(0,120));
  await until(()=>Array.from(document.querySelectorAll('#studio-tools tr[data-image] .studio-stages-done b.done')).some(item=>item.textContent==='T'));
  assert(true,'The list marks the image\'s tissue as done (green T), without reloading');
  assert(state.viewer===viewer,'The image stays open throughout');
  const name=document.querySelector('#studio-tools tr[data-image] button.studio-item-name');name.click();
  await until(()=>state.layers.some(item=>item.id==='drawn-artifacts') && state.layers.some(item=>item.id==='tissue-regions' && item.editable));
  assert(true,'Opening an image from the list shows its tissue and its drawn artifacts layer');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}
},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300)
