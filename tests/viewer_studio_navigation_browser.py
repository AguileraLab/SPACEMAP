"""Late tab responses must not replace controls or overlays in a newer tab."""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1600;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 const step=id=>document.querySelector('[data-step='+id+']').click();
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab5]'));
 const original=window.fetch;let release,waiting=false;
 window.fetch=async(url,options)=>{
  const response=await original(url,options);
  if(String(url).endsWith('/segmentation_settings') && !options?.method){waiting=true;await new Promise(r=>release=r);}
  return response;
 };
 step('tab5');await until(()=>waiting);step('tab4');release();await delay(150);
 assert(!document.querySelector('#panel .role-toggle'),'A late Segmentation load cannot put N/M controls back in Annotation');
 assert(document.querySelector('[data-step=tab4]').getAttribute('aria-current')==='step','The newer tab stays selected');
 window.fetch=original;step('tab5');await until(()=>tools().querySelector('[aria-label="Cellpose model"]'));
 // Delay a representative successful response, without running a model: this
 // check isolates navigation, not Cellpose accuracy (covered in its own test).
 waiting=false;
 window.fetch=async(url,options)=>{
  if(String(url).endsWith('/segment_preview')){
   waiting=true;await new Promise(r=>release=r);
   return new Response(JSON.stringify({outlines:[[[0,0],[30,0],[30,30],[0,0]]],bounds:[0,0,30,30],count:1,seconds:0.1,device:'CPU'}),{headers:{'Content-Type':'application/json'}});
  }return original(url,options);
 };
 button(tools(),'Test on this view').click();await until(()=>waiting);step('tab4');release();await delay(150);
 assert(!s.previews?.seg && !document.querySelector('#shapes .seg-preview-box'),'A late Segmentation preview cannot draw over a different tab');
 window.fetch=original;
 // Saved settings, rather than defaults, return in the other core forms.
 step('tab5');await until(()=>tools().querySelector('[aria-label="Cell diameter"]'));
 const diameter=tools().querySelector('[aria-label="Cell diameter"]');diameter.value='27';diameter.dispatchEvent(new Event('change'));
 await until(async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/segmentation_settings')).json()).values.tab4_diameter===27);  // saved by itself
 step('tab1');step('tab5');await until(()=>tools().querySelector('[aria-label="Cell diameter"]'));
 assert(tools().querySelector('[aria-label="Cell diameter"]').value==='27','Saved Segmentation settings survive leaving and returning');
 step('tab4');await until(()=>document.querySelector('.studio-stage-tabs'));
 button(document.querySelector('.studio-stage-tabs'),'Marker regions').click();
 await until(()=>button(tools(),'Preview'));
 waiting=false;
 window.fetch=async(url,options)=>{
  if(String(url).endsWith('/marker_regions') && options?.body && JSON.parse(options.body).preview){
   waiting=true;await new Promise(r=>release=r);
   return new Response(JSON.stringify({outlines:[[[0,0],[30,0],[30,30],[0,0]]],regions:1,area_um2:450,level_microns:1,ms:1}),{headers:{'Content-Type':'application/json'}});
  }return original(url,options);
 };
 button(tools(),'Preview').click();await until(()=>waiting);
 button(document.querySelector('.studio-stage-tabs'),'Tissue').click();release();await delay(150);
 assert(!s.previews?.['marker-regions-preview'],'A late Marker regions preview cannot draw over another Annotation stage');
 window.fetch=original;
 assert(s.imagesReady,'Switching steps keeps the image open');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=120)
