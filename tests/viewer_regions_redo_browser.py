"""Browser-check: a Marker regions result deleted in Layers and detected again shows again, also after reopening.

Usage: PYTHONPATH=src:tests python tests/viewer_regions_redo_browser.py
"""

from viewer_studio_regions_browser import setup
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent('page error: '+event.message+' '+event.filename+':'+event.lineno)));
window.confirm=()=>true;
setTimeout(async()=>{
 try{
  const state=window.__state, tools=()=>document.querySelector('#studio-tools');
  const stage=async name=>{await until(()=>Array.from(document.querySelectorAll('.studio-stage-tabs button')).some(item=>item.textContent===name));
    Array.from(document.querySelectorAll('.studio-stage-tabs button')).find(item=>item.textContent===name).click();};
  const row=text=>Array.from(document.querySelectorAll('.studio-runbar .row')).find(item=>item.querySelector('.lbl')?.textContent===text);
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=value;input.dispatchEvent(new Event('change'));};
  const open=()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab4]'));
  document.querySelector('[data-step=tab4]').click();
  await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
  open();
  await until(()=>state.imagesReady && state.layers.some(item=>item.type==='image' && item.info?.channels?.includes('PanCK')) && !state.commandBusy);
  await stage('Marker regions');
  await until(()=>button(tools(),'Add a marker'));
  button(tools(),'Add a marker').click();
  set('Marker 1','PanCK');set('Level of marker 1','100');
  set('Work within','image');set('Smoothing sigma','0');set('Region scale','1');set('Smallest region in square micrometres','10');set('Name the regions','Endothelia');
  await until(()=>!tools().querySelector('.annotation-scope [role=status]')?.textContent);
  const result=()=>state.layers.find(item=>item.name==='Marker regions · Endothelia');
  const ready=()=>{const bar=row('Detect regions');return !!bar&&!button(bar,'This image').disabled;};  // the bar may still be drawing
  const detect=async()=>{await until(ready);button(row('Detect regions'),'This image').click();await until(()=>result()?.features?.length>0&&!state.commandBusy&&ready());};
  await detect();
  state.layerGroups.annotations=true;buildPanel();
  const named=()=>document.querySelector('#panel [data-annotation-name="Endothelia"]');
  await until(()=>named());
  named().querySelector('[aria-label="Delete Endothelia"]').click();
  await until(()=>!result()&&!named());
  assert(true,'Deleting Endothelia in Layers removes its layer');
  await until(ready);
  button(row('Detect regions'),'This image').click();
  await until(()=>!state.commandBusy&&ready()&&result()?.features?.length>0);
  await until(()=>named());
  assert(result().visible!==false,'Detecting Endothelia again shows it again');
  document.querySelector('[data-step=tab5]').click();await delay(300);
  document.querySelector('[data-step=tab4]').click();
  await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
  open();
  await until(()=>!state.commandBusy&&state.imagesReady&&result());
  assert(true,'It is still there after the image is opened again');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(document.querySelector('#panel')?.textContent||'').slice(0,500)));}
},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=240, setup=setup)
