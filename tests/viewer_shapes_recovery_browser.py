"""Lost save responses, explicit recovery and shape-load errors in a disposable Studio."""

from viewer_studio_brush_browser import setup
from viewer_threshold_browser import main


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;};
setTimeout(async()=>{try {
 const s=window.__state;
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>document.querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 document.querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 await until(()=>tissue()?.revision && tissue().features?.length===1 && !tissue().loadingShapes);
 const layer=tissue(), viewer=s.viewer;
 assert(layer.status==='1 polygon loaded.','Opening shows a completed load, not Saving');
 const originalFetch=window.fetch.bind(window), originalTimer=window.setTimeout.bind(window);
 let savedResponse=false;
 // The server saves successfully, but its reply is lost. Shorten only the
 // polygon-request deadline so this recovery test does not take 30 seconds.
 window.setTimeout=(fn,ms,...args)=>originalTimer(fn,ms===30000||ms===120000?500:ms,...args);  // reads wait 30 s, saves 2 minutes
 window.fetch=async(url,options)=>{
   if(String(url).endsWith('/shapes/'+layer.token+'/save')) {
     await originalFetch(url,options);savedResponse=true;
     return new Promise((resolve,reject)=>{
       const fail=()=>reject(new DOMException('Lost response','AbortError'));
       if(options.signal.aborted)fail();else options.signal.addEventListener('abort',fail,{once:true});
     });
   }
   return originalFetch(url,options);
 };
 layer.features[0].properties.named=true;
 layer.features[0].properties.name=layer.features[0].properties.label='Reviewed tissue';
 saveShapes(layer);
 try{await flushShapeSave(layer);}catch(_){}
 window.fetch=originalFetch;window.setTimeout=originalTimer;
 assert(savedResponse && layer.savePending && !layer.saveBusy && layer.status.startsWith('Save not confirmed:'),
   'A lost reply stops Saving, retains the draft and explains recovery');
 buildPanel();refreshRegionList();
 const reload=()=>document.querySelector('[data-shape-reload="tissue-regions"]');
 assert(reload() && !reload().disabled,'Reload saved is available after a timed-out save');
 window.confirm=()=>false;reload().click();await delay(100);
 assert(layer.savePending,'Cancelling recovery keeps the waiting changes');
 window.confirm=()=>true;reload().click();
 await until(()=>!layer.savePending && !layer.loadingShapes);
 assert(layer.features[0].properties.name==='Reviewed tissue' && layer.status.includes('loaded'),
   'Reload finds the save already completed and clears the waiting state');
 const before=JSON.stringify(layer.features), version=layer.revision;
 window.fetch=async(url,options)=>{
   if(String(url).endsWith('/shapes/'+layer.token))throw Error('Connection interrupted');
   return originalFetch(url,options);
 };
 await reloadSavedShapes(layer);window.fetch=originalFetch;
 assert(JSON.stringify(layer.features)===before && layer.revision===version && layer.status.includes('Could not load'),
   'A failed reload preserves the displayed polygons and explains the problem');
 await reloadSavedShapes(layer);
 assert(!layer.shapesLoadError && layer.features.length===1 && s.viewer===viewer,
   'Retrying reload clears the error without reopening the image');
 let active=0,peak=0;
 window.fetch=async(url,options)=>{
   if(String(url).endsWith('/shapes/'+layer.token+'/save')) {
     active++;peak=Math.max(peak,active);
     try{await delay(60);return await originalFetch(url,options);}finally{active--;}
   }
   return originalFetch(url,options);
 };
 layer.features[0].properties.name='First name';saveShapes(layer);
 const first=flushShapeSave(layer);
 await until(()=>active===1);
 layer.features[0].properties.name='Latest name';saveShapes(layer);
 await Promise.all([first,flushShapeSave(layer),flushShapeSave(layer)]);
 window.fetch=originalFetch;
 const saved=await(await fetch(api('/shapes/'+layer.token))).json();
 assert(peak===1 && saved.features[0].properties.name==='Latest name' && !layer.savePending,
   'Several waiting actions save in order with one request at a time');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=120, setup=setup)
