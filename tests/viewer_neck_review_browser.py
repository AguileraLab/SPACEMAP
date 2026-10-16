"""Review several necks sequentially, including save/reload and undo."""
from viewer_studio_regions_browser import setup as image
from viewer_threshold_browser import main


def setup(root):
    import json
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union
    from spacemap_gui.studio import annotation
    image(root)
    output = root / 'outputs/processed_annotations/stage0_tissue_annotation'
    path = annotation.valid_tissue(output, 'markers')
    path.parent.mkdir(parents=True, exist_ok=True)
    tissue = unary_union([box(10, 40, 115, 320), box(145, 40, 250, 320), box(280, 40, 385, 320),
                          box(110, 140, 150, 160), box(245, 220, 285, 240)])
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {}, 'geometry': mapping(tissue)}]}))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 await until(()=>s.imagesReady&&tissue()?.features?.length===1&&tissue().revision&&tools().querySelector('[aria-label="Use neck tools"]'));
 const enabled=tools().querySelector('[aria-label="Use neck tools"]');enabled.click();
 const set=(name,value)=>{const input=tools().querySelector('[aria-label="'+name+'"]');input.value=value;input.dispatchEvent(new Event('change'));};
 set('Widest neck','30');set('Minimum on each side','1000');
 const ready=()=>!s.brushBusy&&!tissue().saveBusy&&!tissue().savePending;
 const original=JSON.parse(JSON.stringify(tissue().features));
 const preview=async()=>{button(tools(),'Preview').click();await until(()=>s.neckReview?.regions.length===2&&!button(tools(),'Preview').disabled);};
 await preview();
 assert(tools().querySelectorAll('.neck-review-list .row').length===2,'Two detected necks have review rows');
 assert(tools().querySelector('.neck-review').open&&parseFloat(getComputedStyle(tools().querySelector('.neck-review-list')).maxHeight)<=200,'The review list opens with a bounded scrolling height');
 const first=s.neckReview.regions[0],second=s.neckReview.regions[1];
 const previewCount=()=>performance.getEntriesByType('resource').filter(r=>r.name.endsWith('/pieces_preview')).length;
 const count=previewCount();
 button(tools().querySelector('.neck-review-row'),'Approve').click();await until(()=>tissue().features.length===2&&s.neckReview?.regions.length===1&&ready());
 assert(s.neckReview?.regions.length===1&&s.neckReview.regions[0].id===second.id,'Approving one neck keeps the other suggestion');
 assert(document.querySelector('#shapes .seg-preview.neck')&&tools().querySelectorAll('.neck-review-list .row').length===1,'Remaining neck stays red and listed after approval');
 assert(previewCount()===count,'Approval does not rerun detection');
 // Reload the same saved polygons and keep the unfinished review available.
 await loadShapes(tissue());renderShapes();buildPanel();refreshRegionList();
 assert(s.neckReview?.regions.length===1&&document.querySelector('#shapes .seg-preview.neck'),'Reloading unchanged saved tissue keeps the remaining neck visible');
 const row=tools().querySelector('.neck-review-list .row');button(row,'Go').click();
 await delay(350);
 const center=s.viewer.viewport.getCenter();const point=s.layers.find(l=>l.type==='image').item.viewportToImageCoordinates(center);
 assert(Math.abs(point.x-second.x)<20&&Math.abs(point.y-second.y)<20,'Go centres the next neck');
 await s.neckReview.approve(second);await until(ready);
 assert(tissue().features.length===3&&s.neckReview.regions.length===0,'Both cuts can be approved without another Preview');
 assert(previewCount()===count,'Reviewing the full queue uses one detection request');
 undoShapes(tissue());await until(()=>ready()&&s.neckReview?.regions.length===1);
 assert(tissue().features.length===2&&s.neckReview.regions[0].id===second.id,'Undo restores the cut neck to the review list');
 redoShapes(tissue());await until(()=>ready()&&s.neckReview?.regions.length===0);
 assert(tissue().features.length===3,'Redo reapplies the cut and removes only its suggestion');
 await showTissuePreview({features:original,layer:tissue()});
 await until(()=>s.neckReview?.regions.length===2);
 assert(tissue().preview&&tools().querySelectorAll('.neck-review-row').length===2,'Returning to the original geometry restores its neck review');
 const savedRevision=tissue().revision;
 await s.neckReview.approve(s.neckReview.regions[0]);await until(()=>ready()&&s.neckReview?.regions.length===1);
 assert(tissue().preview&&tissue().revision===savedRevision&&document.querySelector('#shapes .seg-preview.neck'),'A preview cut keeps remaining suggestions without writing saved tissue');
 button(tools().querySelector('.neck-review-row'),'Reject').click();
 assert(!s.neckReview.regions.length&&tissue().features.length===2,'Reject dismisses only the suggestion');
 button(tools(),'Show rejected (1)').click();
 assert(s.neckReview.regions.length===1,'Rejected suggestions can be restored');
 button(tools(),'Hide').click();
 assert(!document.querySelector('#shapes .seg-preview.neck')&&tools().querySelectorAll('.neck-review-row').length===1,'Hide removes the overlay but preserves the review list');
 button(tools().querySelector('.neck-review-row'),'Go').click();
 assert(s.neckReview.regions.length===1&&document.querySelector('#shapes .seg-preview.neck'),'Go shows a hidden suggestion again');
 enabled.click();assert(!s.neckReview&&tools().querySelector('.neck-review').hidden,'Switching neck tools off clears the queue and overlay');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+toolsText()));}},100);
function toolsText(){return document.body.textContent.slice(-6000);}
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup)
