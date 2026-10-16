"""Browser checks of tissue editing, artifact review, persistent history and brush speed."""

from viewer_studio_regions_browser import setup as marker_image
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0; const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try {
 const s=window.__state, tools=()=>document.querySelector('#studio-tools');
 const stage=t=>button(document.querySelector('.studio-stage-tabs'),t).click();
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions'), artifacts=()=>s.layers.find(l=>l.id==='artifact-debris');
 await until(()=>tissue()?.revision && tissue().features?.length===1 && s.imagesReady);
 stage('Tissue');
 assert(tissue().editable && tissue().regions && tissue().features[0].properties.name==='tissue_1','Tissue opens as connected editable pieces');
 const initial=tissue().revision, id=tissue().features[0].properties.id;
 const viewer=s.viewer;
 const at=(x,y)=>viewer.viewport.pixelFromPoint(s.layers.find(l=>l.type==='image').item.imageToViewportCoordinates(x,y));
 const click=(x,y)=>viewer.raiseEvent('canvas-click',{quick:true,position:at(x,y),originalEvent:{}});
 click(60,60);
 await until(()=>s.selected?.layer===tissue());
 assert(document.querySelector('#shapes path.selected'),'Selecting tissue highlights its boundary');
 const stroke=async(mode,from,to,radius)=>{
   s.brushRadius=radius;
   const originalEvent=mode==='add'?{shiftKey:true}:{altKey:true};
   const start=performance.now();
   viewer.raiseEvent('canvas-press',{position:at(...from),originalEvent});
   assert(!!s.brushStroke,'Modifier press starts a brush stroke');
   const before=performance.getEntriesByType('resource').filter(r=>/\/brush/.test(r.name)).length;
   viewer.raiseEvent('canvas-drag',{position:at(...to),originalEvent});
   assert(document.querySelector('#annotation-brush').className===mode && document.querySelector('.brush-trail'),'Brush colour and stroke preview appear immediately');
   assert(performance.getEntriesByType('resource').filter(r=>/\/brush/.test(r.name)).length===before,'Dragging sends no geometry requests');
   viewer.raiseEvent('canvas-release',{position:at(...to),originalEvent});
   await until(()=>!s.brushBusy);
   return performance.now()-start;
 };
 const elapsed=await stroke('erase',[200,100],[200,240],6);
 assert(tissue().features.length===2 && tissue().features.every(f=>f.geometry.type==='Polygon'),'Erasing the neck keeps both tissue pieces');
 assert(s.viewer===viewer,'Brush editing keeps the image and zoom viewer');
 assert(elapsed<2500,'Completed stroke under 2.5 seconds ('+Math.round(elapsed)+' ms)');
 const erased=tissue().revision;
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
 await until(()=>tissue().features.length===1 && !tissue().savePending && !tissue().saveBusy);
 assert(tissue().features[0].properties.id===id,'Undo restores the original tissue piece');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'y',ctrlKey:true,bubbles:true}));
 await until(()=>tissue().features.length===2 && !tissue().savePending && !tissue().saveBusy);
 assert(true,'Redo restores the cut');
 await delay(550); click(60,60);
 const size=s.brushRadius;
 const scroll={position:at(60,60),scroll:1,originalEvent:{shiftKey:true}};
 viewer.raiseEvent('canvas-scroll',scroll);
 assert(s.brushRadius>size && scroll.preventDefaultAction,'Modifier-wheel resizes the brush without zooming');
 await stroke('add',[60,60],[20,60],10);
 assert(tissue().features.length===2,'Painting grows the selected piece and preserves the other');
 stage('Artifacts');
 assert(tissue().visible && s.viewer===viewer,'Tissue stays visible when switching stages');
 await until(()=>button(tools(),'Subtract from tissue'));
 assert(button(tools(),'Subtract from tissue'),'Subtracting from the tissue is its own action, after review');
 stage('Tissue');
 await until(()=>tools().querySelector('.annotation-history-box'));
 tools().querySelector('.annotation-history-box').open=true;
 await until(()=>tools().querySelectorAll('.annotation-history .row').length>=3);
 const beforeRow=Array.from(tools().querySelectorAll('.annotation-history .row')).find(r=>r.textContent.includes('Before the first change'));
 button(beforeRow,'Restore').click();
 await until(()=>tissue().features.length===1 && /Restored/.test(tissue().status||''));
 assert(true,'History restores the original tissue after multiple strokes');
 const history=await(await fetch('/spacemap-viewer/api/shapes/'+tissue().token+'/history')).json();
 assert(history.entries.some(e=>e.after===erased) && history.entries[0].label==='Restored an earlier version','Restoring keeps the cut and painted versions available');
 // A stale tab is rejected instead of overwriting newer work.
 const conflict=await fetch('/spacemap-viewer/api/shapes/'+tissue().token+'/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:initial,features:[]})});
 assert(conflict.status===409,'An older window cannot overwrite the tissue');
 // The two polygon lists share selection, including their partial select-all state.
 tissue().open=true;
 stage('Tissue');
 await until(()=>tools().querySelector('.region-selection input'));
 const allBoxes=()=>Array.from(document.querySelectorAll('[aria-label="Select all polygons in tissue"]'));
 document.querySelectorAll('#panel details.layer-group').forEach(d=>{d.open=true;});  // Annotations start folded in Layers
 s.annotationRowsOpen={...(s.annotationRowsOpen||{}),['::'+tissue().name]:true};buildPanel();  // the tissue's name row, open
 await until(()=>allBoxes().length===2);
 allBoxes()[0].click();
 assert(allBoxes().every(b=>b.checked),'Select all in either panel selects polygons in both panels');
 assert(document.querySelectorAll('.region-list input[data-region-id]:checked').length>=2,'Each polygon has a selected checkbox');
 button(tools(),'Delete selected').click();
 await until(()=>tissue().features.length===0 && !tissue().savePending && !tissue().saveBusy);
 assert(allBoxes().every(b=>!b.checked && b.disabled),'Delete selected removes every polygon and clears both select-all controls');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
 await until(()=>tissue().features.length===1 && !tissue().savePending && !tissue().saveBusy);
 assert(true,'Undo brings back polygons deleted with Delete selected');
 // Rerun detection replaces the edited result after confirmation, and retains it in History.
 let confirmations=0; window.confirm=()=>{confirmations++;return true;};
 await until(()=>tools().querySelector('[aria-label="Tissue mode"]') && Array.from(document.querySelectorAll('.studio-runbar .row')).some(r=>r.querySelector('.lbl')?.textContent==='Detect tissue'));
 const mode=tools().querySelector('[aria-label="Tissue mode"]');mode.value='codex_only';mode.dispatchEvent(new Event('change'));
 const oldRevision=tissue().revision;
 const runRow=Array.from(document.querySelectorAll('.studio-runbar .row')).find(r=>r.querySelector('.lbl')?.textContent==='Detect tissue');
 button(runRow,'This image').click();
 await until(()=>tissue().revision!==oldRevision && tissue().features.length>0);
 const detected=await(await fetch('/spacemap-viewer/api/shapes/'+tissue().token+'/history')).json();
 assert(confirmations===1 && detected.entries[0].label==='Detected tissue','Rerun asks before replacing edited tissue');
 assert(detected.entries.some(e=>e.before===oldRevision) && s.viewer===viewer,'Rerun keeps the earlier tissue in History and updates the open layer');
 // Subtracting artifacts is checked by viewer_artifact_review_browser.py.
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,stroke_ms:Math.round(elapsed)})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#message')?.textContent));}},100);
</script>
"""


def setup(root):
    import json
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union
    from spacemap_gui.studio import annotation

    marker_image(root)
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    path = annotation.valid_tissue(output, "markers")
    path.parent.mkdir(parents=True, exist_ok=True)
    tissue = unary_union([box(30, 30, 170, 320), box(160, 155, 250, 185), box(240, 30, 380, 320)])
    path.write_text(json.dumps({"type":"FeatureCollection", "features":[{"type":"Feature", "properties":{}, "geometry":mapping(tissue)}]}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=240, setup=setup)
