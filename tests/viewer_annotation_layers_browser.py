"""Repeated Annotation openings/previews/runs, legacy duplicates and saved-file clearing."""

from viewer_studio_brush_browser import setup
from viewer_threshold_browser import main


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<2400;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
setTimeout(async()=>{try{
 const s=window.__state, tools=()=>document.querySelector('#studio-tools');
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 const ids=['tissue-regions','drawn-artifacts','marker-regions'];
 const unique=()=>ids.every(id=>s.layers.filter(l=>l.id===id).length===1);
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 const open=()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 open();await until(()=>tissue()?.revision && unique() && !s.commandBusy);
 const spec=JSON.parse(JSON.stringify(s.spec));
 const copies=spec.layers.filter(l=>ids.includes(l.id)).flatMap(l=>Array.from({length:3},()=>({id:l.id,type:l.type,name:l.name,token:l.token,image_token:l.image_token,added:true})));
 localStorage.setItem('spacemap-viewer-view:'+spec.id,JSON.stringify({extras:copies,layers:{},order:copies.map(l=>l.id)}));
 await init(spec);await until(()=>s.imagesReady);
 assert(unique() && tissue().regions && tissue().annotation_kind==='tissue','An old remembered view with nine copies opens with one complete layer of each kind');
 await delay(650);
 assert(!loadView(spec.id).extras.some(l=>ids.includes(l.id)),'Workflow layers are not remembered a second time as extra layers');
 const viewer=s.viewer;
 for(let i=0;i<3;i++){open();await until(()=>!s.commandBusy);await s.commandQueue;assert(unique() && s.viewer===viewer,'Reopening the same image keeps one Tissue layer and the current viewer');}
 button(document.querySelector('.studio-stage-tabs'),'Tissue').click();
 await until(()=>tools().querySelector('[aria-label="Tissue mode"]'));
 const mode=tools().querySelector('[aria-label="Tissue mode"]');mode.value='codex_only';mode.dispatchEvent(new Event('change'));
 const stored=await(await fetch(api('/shapes/'+tissue().token))).json();
 const row=()=>Array.from(document.querySelectorAll('.studio-runbar .row')).find(r=>r.querySelector('.lbl')?.textContent==='Detect tissue');
 const jobs=async()=>(await(await fetch(api('/studio/browser-check/jobs'))).json()).jobs;
 window.confirm=()=>true;  // a rerun asks before replacing the saved tissue
 for(let i=0;i<2;i++){
  const said=()=>document.querySelector('.studio-runbar').textContent;
  document.querySelectorAll('.studio-runbar [role=status]').forEach(n=>{n.textContent='';});
  await until(()=>!button(row(),'This image').disabled);
  button(row(),'This image').click();
  await until(()=>/Saved \d+ tissue polygons/.test(said()) && tissue().features.length>0 && !s.commandBusy && !tissue().saveBusy);
  assert(true,'Detect tissue saves the tissue at once');
  assert(unique() && tissue().editable && !tissue().preview && s.viewer===viewer,'Detection and rerun replace polygons in the existing layer without moving the image');
 }
 assert(tissue().saved_file.endsWith('/markers_valid_tissue.geojson') && tools().textContent.includes(tissue().saved_file),'Saved file shows the exact annotation path');
 const count=tissue().features.length;
 button(tools(),'Clear saved polygons').click();
 await until(()=>!tissue().features.length && !tissue().savePending && !tissue().saveBusy);
 const empty=await(await fetch(api('/shapes/'+tissue().token))).json();
 assert(!empty.features.length && unique(),'Clear saved polygons empties the file and keeps just one layer');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
 await until(()=>tissue().features.length===count && !tissue().savePending && !tissue().saveBusy);
 assert(true,'Undo restores the cleared polygons');
 const layer=tissue(), revision=layer.revision;
 removeLayer(layer);
 assert(!tissue() && !document.querySelector('#shapes [data-layer="tissue-regions"]'),'Removing a layer also clears its outlines from the image');
 const kept=await(await fetch(api('/shapes/'+layer.token))).json();
 assert(kept.revision===revision && kept.features.length===count,'Removing a layer leaves the saved file unchanged');
 await Promise.all([addLayerToView({...spec.layers.find(l=>l.id==='tissue-regions')}),addLayerToView({...spec.layers.find(l=>l.id==='tissue-regions')})]);
 assert(unique() && tissue().features.length===count,'Overlapping requests to add the same layer still produce one complete layer');
 const savedBeforeRestore=JSON.stringify(tissue().features);
 tools().querySelector('.annotation-history-box').open=true;
 await until(()=>Array.from(tools().querySelectorAll('.annotation-history .row')).some(r=>r.textContent.includes('Before the first change')));
 const first=Array.from(tools().querySelectorAll('.annotation-history .row')).find(r=>r.textContent.includes('Before the first change'));
 button(first,'Restore').click();
 await until(()=>!tissue().preview && tissue().status.includes('Restored'));
 assert(tissue().editable && unique(),'Restoring History returns to the earlier saved polygons');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
 await until(()=>!tissue().savePending && !tissue().saveBusy && JSON.stringify(tissue().features)===savedBeforeRestore);
 assert(true,'Undo after restoring History brings back the polygons from before');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+toolsText()));}},100);
function toolsText(){return document.querySelector('#studio-tools')?.textContent||'';}
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=240, setup=setup)
