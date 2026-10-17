"""Browser-check a save the page stopped waiting for (a slow server) that still reached the file: the next edit
carries on from it instead of being refused as a change from another window.

Usage: PYTHONPATH=src:tests python tests/viewer_late_save_browser.py
"""
from viewer_threshold_browser import main
from viewer_measure_regions_browser import setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 await until(()=>s.imagesReady&&tissue()?.features?.length===3&&tools().querySelector('.region-list input[type=checkbox]'));
 const read=async()=>(await (await fetch('/spacemap-viewer/api/shapes/'+tissue().token)).json());
 const tick=i=>tools().querySelectorAll('.region-list input[type=checkbox]')[i].click();
 // The server writes the save, but its reply never reaches the page (as when the page gives up waiting).
 const originalFetch=window.fetch;let late=0;
 window.fetch=async(url,options)=>{
  const response=await originalFetch(url,options);
  if(String(url).endsWith('/save')&&options?.method==='POST'&&!late++)throw new DOMException('slow','AbortError');
  return response;
 };
 tick(0);button(tools(),'Delete selected').click();
 await until(()=>tissue().saveError&&/did not reply/.test(tissue().saveError));
 assert((await read()).features.length===2,'The save reached the file although the page stopped waiting');
 window.fetch=originalFetch;
 await until(()=>tools().querySelectorAll('.region-list input[type=checkbox]').length===2);
 tick(0);button(tools(),'Delete selected').click();
 await until(()=>!tissue().savePending&&!tissue().saveBusy&&!tissue().saveError);
 assert((await read()).features.length===1&&!/changed at/.test(tissue().status||''),'The next edit carries on from the late save instead of being refused');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+(s=window.__state,s.layers.find(l=>l.id==='tissue-regions')?.status)));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=120, setup=setup)
