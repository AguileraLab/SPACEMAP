"""Modifier drawing, precise split/merge and individual neck review on disposable tissue."""
from viewer_studio_brush_browser import setup
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0; const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state, tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 await until(()=>s.imagesReady&&tissue()?.features?.length===1&&tissue().revision);
 await until(()=>button(tools(),'Preview'));
 assert(!button(document.querySelector('.studio-stage-tabs'),'Pieces')&&tools().textContent.includes('Necks and pieces'),'Neck tools are inside Tissue, with no separate Pieces tab');
 const enabled=tools().querySelector('[aria-label="Use neck tools"]');
 assert(!enabled.checked,'Neck tools start off and do no automatic analysis');enabled.click();
 const viewer=s.viewer, at=(x,y)=>viewer.viewport.pixelFromPoint(s.layers.find(l=>l.type==='image').item.imageToViewportCoordinates(x,y));
 const settled=()=>!s.brushBusy&&!tissue().savePending&&!tissue().saveBusy;
 const pick=()=>{s.selected=null;s.picks={layer:tissue(),ids:new Set([tissue().features[0].properties.id])};};
 const undo=async()=>{undoShapes(tissue());await until(settled);};
 setTool('rect');
 viewer.viewport.zoomBy(3);viewer.viewport.applyConstraints(true);
 const beforeFit=tissue().revision;
 const wheelDown=new MouseEvent('mousedown',{button:1,bubbles:true,cancelable:true});viewer.canvas.dispatchEvent(wheelDown);
 const wheelClick=new MouseEvent('auxclick',{button:1,bubbles:true,cancelable:true});viewer.canvas.dispatchEvent(wheelClick);
 await until(()=>Math.abs(viewer.viewport.getZoom()/viewer.viewport.getHomeZoom()-1)<0.01);
 assert(wheelDown.defaultPrevented&&wheelClick.defaultPrevented,'Mouse-wheel click fits the image without browser auto-scroll');
 assert(tissue().revision===beforeFit&&!drawing.start&&!s.brushStroke,'Mouse-wheel click in Rectangle mode creates no annotation');
 let worst=0;
 for(const tool of ['off','rect','polygon','freehand']){
   setTool(tool);pick();s.brushRadius=6;
   const event={altKey:true}, start=performance.now(), count=tissue().features.length;
   viewer.raiseEvent('canvas-press',{position:at(200,100),originalEvent:event});
   assert(!!s.brushStroke,tool+': erase starts from a checked polygon');
   viewer.raiseEvent('canvas-drag',{position:at(200,240),originalEvent:event});
   assert(document.querySelector('#annotation-brush').className==='erase'&&!document.querySelector('#annotation-brush').hidden,tool+': round erase cursor is visible');
   viewer.raiseEvent('canvas-release',{position:at(200,240),originalEvent:event});
   await until(settled);worst=Math.max(worst,performance.now()-start);
   assert(tissue().features.length===count+1&&!drawing.points,tool+': neck erased without creating an extra drawing');
   await undo();pick();
   const before=JSON.stringify(tissue().features), e={shiftKey:true};
   viewer.raiseEvent('canvas-press',{position:at(60,60),originalEvent:e});
   viewer.raiseEvent('canvas-drag',{position:at(20,60),originalEvent:e});
   viewer.raiseEvent('canvas-release',{position:at(20,60),originalEvent:e});await until(settled);
   assert(JSON.stringify(tissue().features)!==before&&tissue().features.length===1,tool+': paint grows only the chosen polygon');
   await undo();
 }
 assert(worst<2500,'Saved erase strokes finish under 2.5 seconds (worst '+Math.round(worst)+' ms)');
 pick();setTool('freehand');s.shiftDown=true;moveBrushCursor({position:at(60,60)});
 const radius=s.brushRadius, e={position:at(60,60),scroll:1,originalEvent:{shiftKey:true}};brushScroll(e);
 assert(s.brushRadius>radius&&e.preventDefaultAction,'Wheel resizes the brush without zooming');s.shiftDown=false;
 viewer.raiseEvent('canvas-press',{position:at(60,60),originalEvent:{altKey:true}});
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
 const rev=tissue().revision;viewer.raiseEvent('canvas-drag',{position:at(200,100)});viewer.raiseEvent('canvas-release',{position:at(200,100)});await delay(100);
 assert(tissue().revision===rev&&!drawing.points,'Escape cancels the brush without creating a freehand drawing');
 setTool('rect');s.drawInto='tissue-regions';const n=tissue().features.length;
 viewer.raiseEvent('canvas-press',{position:at(4,4)});viewer.raiseEvent('canvas-drag',{position:at(20,20)});viewer.raiseEvent('canvas-release',{position:at(20,20)});
 await until(()=>tissue().features.length===n+1&&settled());assert(true,'Ordinary Rectangle drawing still adds a polygon');await undo();
 pick();setTool('split');
 assert(document.querySelector('[aria-label="Cut width"]').value==='2','Split defaults to a 2 full-image-pixel cut');
 viewer.raiseEvent('canvas-press',{position:at(200,100)});viewer.raiseEvent('canvas-drag',{position:at(200,240)});viewer.raiseEvent('canvas-release',{position:at(200,240)});
 await until(()=>tissue().features.length===2&&settled());assert(true,'Split cuts the selected neck into two pieces');
 setTool('merge');
 assert(!document.querySelector('#merge-selected').disabled,'Split pieces are selected and ready to merge');
 const gap=document.querySelector('[aria-label="Maximum gap"]');gap.value='1';gap.dispatchEvent(new Event('change'));
 const splitRevision=tissue().revision;document.querySelector('#merge-selected').click();await until(()=>!s.brushBusy);
 assert(tissue().features.length===2&&tissue().revision===splitRevision,'Too-small maximum gap refuses merge without changing tissue');
 gap.value='3';gap.dispatchEvent(new Event('change'));document.querySelector('#merge-selected').click();await until(()=>tissue().features.length===1&&settled());
 assert(true,'Custom maximum gap merges the selected pieces');await undo();
 // A newly drawn bridge can be joined with both existing tissue polygons.
 setTool('rect');s.drawInto='tissue-regions';
 viewer.raiseEvent('canvas-press',{position:at(190,160)});viewer.raiseEvent('canvas-drag',{position:at(210,180)});viewer.raiseEvent('canvas-release',{position:at(210,180)});
 await until(()=>tissue().features.length===3&&settled());
 s.picks={layer:tissue(),ids:new Set(tissue().features.map(f=>f.properties.id))};setTool('merge');
 document.querySelector('#merge-selected').click();await until(()=>settled()&&tissue().features.length===1);
 assert(true,'Merge joins two saved polygons and a drawn bridge without a missing-route error');
 await undo();assert(tissue().features.length===3,'Undo restores both polygons and the drawn bridge');await undo();await undo();
 // A preview is edited locally, until Save polygons is explicitly used.
 const stored=tissue().revision, features=JSON.parse(JSON.stringify(tissue().features));
 await showTissuePreview({features,layer:tissue()});pick();setTool('polygon');
 viewer.raiseEvent('canvas-press',{position:at(200,100),originalEvent:{altKey:true}});viewer.raiseEvent('canvas-drag',{position:at(200,240),originalEvent:{altKey:true}});viewer.raiseEvent('canvas-release',{position:at(200,240),originalEvent:{altKey:true}});
 await until(()=>!s.brushBusy&&tissue().features.length===2);
 const saved=await(await fetch('/spacemap-viewer/api/shapes/'+tissue().token)).json();
 assert(tissue().preview&&saved.revision===stored&&saved.features.length===1,'Preview brushing never writes into saved tissue');await undo();
 setTool('off');button(tools(),'Preview').click();await until(()=>s.neckReview?.regions.length&&!button(tools(),'Preview').disabled);
 assert(s.previews['pieces-necks']&&document.querySelector('#shapes .seg-preview.neck'),'Detected necks have visible red polygons');
 setVisible(tissue(),false);assert(!document.querySelector('#shapes .seg-preview.neck'),'Hiding Tissue hides the neck preview too');
 setVisible(tissue(),true);assert(document.querySelector('#shapes .seg-preview.neck'),'Showing Tissue restores the neck preview');
 enabled.click();assert(!document.querySelector('#shapes .seg-preview.neck')&&!s.previews['pieces-kept'],'Switching neck tools off clears all of their previews immediately');
 enabled.click();button(tools(),'Preview').click();await until(()=>s.neckReview?.regions.length&&!button(tools(),'Preview').disabled);
 const openMenu=neck=>{const p=at(neck.x,neck.y),b=viewer.element.getBoundingClientRect();viewer.canvas.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:b.left+p.x,clientY:b.top+p.y}));};
 const neck=s.neckReview.regions[0];openMenu(neck);
 assert(button(document.querySelector('#ctxmenu'),'Approve cut')&&button(document.querySelector('#ctxmenu'),'Reject suggestion'),'Right-click a neck offers approve and reject');
 button(document.querySelector('#ctxmenu'),'Reject suggestion').click();
 assert(!s.neckReview.regions.length&&tissue().features.length===1,'Reject hides the neck without cutting tissue');
 button(tools(),'Show rejected (1)').click();assert(s.neckReview.regions.length===1,'Rejected suggestions can be brought back');
 openMenu(neck);button(document.querySelector('#ctxmenu'),'Approve cut').click();await until(()=>!s.brushBusy&&tissue().features.length===2);
 assert(tissue().preview&&tissue().revision===stored,'Approving a preview neck cuts only the preview');await undo();
 clearShapePreview(tissue());setTool('off');button(tools(),'Preview').click();await until(()=>s.neckReview?.regions.length&&!button(tools(),'Preview').disabled);
 const previewRequest=performance.getEntriesByType('resource').filter(r=>r.name.endsWith('/pieces_preview')).at(-1);
 assert(!!previewRequest,'Neck preview succeeds through the current Tissue panel');
 const savedNeck=s.neckReview.regions[0];openMenu(savedNeck);button(document.querySelector('#ctxmenu'),'Approve cut').click();await until(()=>settled()&&tissue().features.length===2);
 assert(tissue().revision!==stored,'Approving a saved neck writes the cut and updates the polygon list');await undo();
 const minimum=tools().querySelector('[aria-label="Minimum on each side"]');minimum.value='100000';minimum.dispatchEvent(new Event('change',{bubbles:true}));
 button(tools(),'Preview').click();await until(()=>/0 necks to review/.test(tools().textContent));assert(!s.neckReview?.regions.length,'Both-side size requirement prevents cutting a small lobe');
 button(document.querySelector('.studio-stage-tabs'),'Artifacts').click();await until(()=>tools().querySelector('[aria-label="Artifact scale"]'));
 button(document.querySelector('.studio-stage-tabs'),'Tissue').click();await until(()=>tools().querySelector('[aria-label="Minimum on each side"]'));
 assert(tools().querySelector('[aria-label="Minimum on each side"]').value==='100000','Neck settings survive moving between stages');
 const toggle=tools().querySelector('[aria-label="Use neck tools"]');if(toggle.checked)toggle.click();
 button(document.querySelector('.studio-stage-tabs'),'Artifacts').click();await until(()=>tools().querySelector('[aria-label="Artifact scale"]'));
 button(document.querySelector('.studio-stage-tabs'),'Tissue').click();await until(()=>tools().querySelector('[aria-label="Use neck tools"]'));
 assert(!tools().querySelector('[aria-label="Use neck tools"]').checked,'Neck tools stay off after changing stages');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,worst_stroke_ms:Math.round(worst)})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#message')?.textContent));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup)
