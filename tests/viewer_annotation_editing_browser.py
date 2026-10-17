"""Annotation settings during navigation/saves, and pruning previews without data loss."""

from viewer_studio_brush_browser import setup as base_setup
from viewer_threshold_browser import main


def setup(root):
    import numpy as np
    import tifffile

    base_setup(root)
    planes = np.zeros((3, 400, 400), dtype=np.uint16)
    for start in (15, 145, 275):
        planes[:, start:start + 110, start:start + 110] = 1000
    tifffile.imwrite(root / "markers.ome.tiff", planes, metadata={"axes": "CYX", "PhysicalSizeX": 1., "PhysicalSizeY": 1.,
                        "Channel": {"Name": ["PanCK", "ECad", "CD3"]}})


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1600;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools'),panel=()=>document.querySelector('#panel [data-layer-id="tissue-regions"]');
 const stage=name=>button(document.querySelector('.studio-stage-tabs'),name).click();
 const input=name=>tools().querySelector('[aria-label="'+name+'"]');
 const set=(name,value)=>{const e=input(name);e.value=value;e.dispatchEvent(new Event('change',{bubbles:true}));};
 const read=path=>fetch(api(path)).then(r=>r.json());
 const settings=()=>read('/studio/browser-check/tissue_settings');
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 const footerSave=()=>tools().querySelector('.studio-runbar [data-save-tissue]');
 const savedChip=()=>footerSave()?.parentElement.querySelector('[role=status]');
 const tissueBadge=()=>tools().querySelector('tr[data-image="markers.ome.tiff"] [data-annotation-stage="tissue"]');
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>tissue()?.revision && !s.commandBusy);
 stage('Tissue');await until(()=>input('Smoothing radius'));
 assert(footerSave()&&!footerSave().disabled&&savedChip().textContent==='Saved'&&tissueBadge().classList.contains('done'),'Tissue has a bottom Save polygons button and confirms already saved tissue');
 set('Tissue mode','codex_only');set('Detection scale','1');set('Smoothing radius','3');set('Smallest piece','5000');set('Fill holes smaller than (µm²)','30');
 assert(input('Smallest piece').closest('.field').textContent.includes('µm²'),'Smallest uses the same visible µm² unit as Fill holes');
 stage('Marker regions');await until(()=>input('Region scale'));
 stage('Tissue');await until(()=>input('Smoothing radius'));
 assert(input('Smoothing radius').value==='3' && input('Detection scale').value==='1' && input('Fill holes smaller than (µm²)').value==='30','Leaving Tissue immediately saves and restores its settings');
 set('Smoothing radius','4');
 document.querySelector('[data-step=tab1]').click();await until(()=>document.querySelector('[data-step=tab1]').getAttribute('aria-current')==='step');
 document.querySelector('[data-step=tab4]').click();await until(()=>input('Smoothing radius'));
 assert(input('Smoothing radius').value==='4','Tissue settings survive moving to another Studio step');
 // A delayed response must not erase a newer edit, or race a navigation save.
 const originalFetch=window.fetch;let pending=false,release=null,posts=0;
 window.fetch=async(url,options)=>{
  if(String(url).endsWith('/tissue_settings') && options?.method==='POST'){
   posts++;const response=await originalFetch(url,options);
   if(posts===1){pending=true;await new Promise(r=>release=r);}return response;
  }return originalFetch(url,options);
 };
 set('Smoothing radius','5');await until(()=>pending);
 set('Smoothing radius','6');stage('Marker regions');release();
 await until(()=>input('Region scale'));window.fetch=originalFetch;
 stage('Tissue');await until(()=>input('Smoothing radius'));
 assert(input('Smoothing radius').value==='6' && (await settings()).values.tissue_morph_radius===6 && posts===2,'The newest edit survives a slow save and an immediate tab switch');
 let fail=true;
 window.fetch=(url,options)=>String(url).endsWith('/tissue_settings') && options?.method==='POST' && fail ? Promise.resolve(new Response('{"error":"temporary test failure"}',{status:503,headers:{'Content-Type':'application/json'}})) : originalFetch(url,options);
 set('Smoothing radius','3');await until(()=>tools().textContent.includes('Could not save settings'));
 stage('Marker regions');await delay(150);
 assert(input('Smoothing radius')?.value==='3','A failed save keeps the values and current tab instead of discarding changes');
 fail=false;button(tools(),'Save').click();await until(async()=>(await settings()).values.tissue_morph_radius===3);window.fetch=originalFetch;
 const stored=await read('/shapes/'+tissue().token);
 // This image detects and saves at once (Ctrl+Z undoes; History keeps the earlier tissue).
 const detectRow=Array.from(tools().querySelectorAll('.studio-runbar .row')).find(r=>r.textContent.startsWith('Detect tissue'));
 button(detectRow,'This image').click();
 await until(async()=>(await read('/shapes/'+tissue().token)).revision!==stored.revision&&tissue().features.length>=3&&!tissue().saveBusy&&!tissue().savePending);
 await until(()=>savedChip().textContent==='Saved'&&tissueBadge().classList.contains('done'));
 assert(true,'This image detects and saves the tissue at once (green T)');
 const count=tissue().features.length;
 assert(tissue().features.every(f=>featureArea(f)*pixelSize()**2>=5000),'Every detected polygon meets the 5,000 µm² minimum');
 const tick=(root,i)=>root.querySelectorAll('.region-list input[type=checkbox]')[i].click();
 tick(tools(),0);tick(tools(),1);
 assert(s.picks.ids.size===2 && !button(tools(),'Delete selected').disabled,'Selecting two polygons enables Delete selected');
 let failPolygons=true,saveCalls=0;
 window.fetch=(url,options)=>{
  if(String(url).endsWith('/shapes/'+tissue().token+'/save')&&options?.method==='POST'){
   saveCalls++;
   if(failPolygons)return Promise.resolve(new Response('{"error":"temporary polygon-save failure"}',{status:503,headers:{'Content-Type':'application/json'}}));
  }
  return originalFetch(url,options);
 };
 const focused=tools().querySelector('.region-list input[type=checkbox]:checked');focused.focus();
 focused.dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true,cancelable:true}));
 assert(tissue().features.length===count-2,'Delete works with focus on a polygon checkbox');
 await until(()=>savedChip().textContent==='Save not confirmed');
 assert(!tissueBadge().classList.contains('done'),'A failed save never shows a green T');
 failPolygons=false;footerSave().click();await until(()=>savedChip().textContent==='Saved'&&!tissue().saveBusy&&!tissue().savePending);
 assert(tissueBadge().classList.contains('done')&&!tissueBadge().classList.contains('waiting'),'Save polygons after a failure saves and turns T green');
 const writes=saveCalls;footerSave().click();await delay(80);
 assert(saveCalls===writes&&savedChip().textContent==='Saved','Saving already-saved tissue confirms it without another write');window.fetch=originalFetch;
 assert((await read('/shapes/'+tissue().token)).features.length===count-2,'The saved file holds exactly the reviewed polygons');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
 await until(()=>tissue().features.length===count&&!tissue().saveBusy&&!tissue().savePending);
 assert(true,'Ctrl+Z brings the deleted polygons back');
 let releasePolygonSave=null,polygonSaving=false;
 window.fetch=async(url,options)=>{
  if(String(url).endsWith('/shapes/'+tissue().token+'/save')&&options?.method==='POST'){
   polygonSaving=true;await new Promise(resolve=>releasePolygonSave=resolve);
  }
  return originalFetch(url,options);
 };
 tick(tools(),0);tick(tools(),1);button(tools(),'Delete selected').click();
 assert(savedChip().textContent==='Not saved'&&tissueBadge().classList.contains('waiting'),'Pending edits immediately mark Tissue unsaved before background saving starts');
 footerSave().click();await until(()=>polygonSaving);
 assert(footerSave().disabled&&savedChip().textContent==='Saving…'&&!tissueBadge().classList.contains('done'),'Footer waits for the save to finish before confirming it');
 releasePolygonSave();window.fetch=originalFetch;
 await until(()=>!tissue().saveBusy && !tissue().savePending);
 assert((await read('/shapes/'+tissue().token)).features.length===count-2,'Multi-delete of saved polygons is written in the background');
 tick(tools(),0);
 const numeric=input('Smoothing radius');numeric.focus();
 const before=tissue().features.length;numeric.dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true}));
 assert(tissue().features.length===before,'Delete while typing a setting does not delete polygons');
 const imageTick=tools().querySelector('.studio-pairs input[type=checkbox]');imageTick.focus();
 imageTick.dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true}));
 assert(tissue().features.length===before,'Delete in the image list does not remove a polygon selected elsewhere');
 assert(s.viewer && s.imagesReady,'The image remains open throughout editing and settings saves');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180, setup=setup)
