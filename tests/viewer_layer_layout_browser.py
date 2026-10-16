"""Layer grouping, output folding and marker alignment with all optional columns."""

from viewer_studio_brush_browser import setup as tissue_project
from viewer_threshold_browser import main


def setup(root):
    tissue_project(root)
    output = root / "outputs/segmentation"
    output.mkdir(parents=True, exist_ok=True)
    for index in range(15):
        (output / f"sample_{index}_cell_boundaries.geojson").write_text('{"type":"FeatureCollection","features":[]}')


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1600;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message)));
setTimeout(async()=>{try{
 const s=window.__state, panel=()=>document.querySelector('#panel'), tools=()=>document.querySelector('#studio-tools');
 const step=id=>document.querySelector('[data-step='+id+']').click();
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 const card=id=>panel().querySelector('[data-layer-id="'+id+'"]');
 const row=()=>panel().querySelector('[data-annotation-name="'+tissue().name+'"]');  // the tissue's name row (Annotations by name)
 const group=key=>panel().querySelector('[data-layer-group="'+key+'"]');
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab4]'));
 step('tab4');await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>tissue()?.features?.length && !s.commandBusy && s.imagesReady);
 document.querySelector('#studio-right [data-tab=layers]').click();
 const image=s.layers.find(l=>l.type==='image'), viewer=s.viewer;
 assert(panel().querySelector('.layer').dataset.layerId===image.id,'The image and its markers come before other layers');
 assert(!group('annotations').open && row().classList.contains('collapsed'),'Annotation groups start folded with compact layer rows');
 const revision=tissue().revision, order=s.layers.map(l=>l.id).join(',');
 group('annotations').querySelector('summary').click();await delay(50);
 row().querySelector('.layer-head .name').click();
 assert(tissue().visible && row().querySelector('.region-list').getBoundingClientRect().height>0,'Opening a layer reveals its polygon list');
 for(const list of document.querySelectorAll('.region-list')) {
   const name=list.querySelector('.region-name'), swatch=list.querySelector('.swatch');
   if(name&&swatch) assert(name.getBoundingClientRect().left-swatch.getBoundingClientRect().right<8&&getComputedStyle(name).textAlign==='left','Piece names sit beside the checkbox and colour in both panels');
 }
 group('annotations').querySelector('summary').click();await delay(50);
 assert(tissue().visible && document.querySelector('#shapes [data-layer="tissue-regions"] path'),'Folding the group keeps its tissue on the image');
 group('annotations').querySelector('summary').click();await delay(50);
 row().querySelector('.layer-head input[type=checkbox]').click();
 assert(!tissue().visible && !document.querySelector('#shapes [data-layer="tissue-regions"]'),'Unticking hides tissue without removing its layer');
 button(panel().querySelector('.layer-filter'),'Shown').click();
 assert(!row(),'Shown filters hidden layers out of the list');
 button(panel().querySelector('.layer-filter'),'All').click();
 assert(!!row(),'All brings hidden layer controls back');
 row().querySelector('.layer-head input[type=checkbox]').click();
 assert(tissue().visible && tissue().revision===revision && order===s.layers.map(l=>l.id).join(',') && s.viewer===viewer,'Showing and grouping preserve saved polygons, drawing order and the viewer');
 const stored=await(await fetch(api('/shapes/'+tissue().token))).json();
 assert(stored.revision===revision,'Show and hide do not write the annotation file');
 group('annotations').open=false;await delay(700);
 const remembered=loadView(s.spec.id);
 assert(remembered.panelLayout===1 && remembered.layerGroups.annotations===false,'The folded group choice is remembered');
 await init(JSON.parse(JSON.stringify(s.spec)));await until(()=>s.imagesReady);
 assert(!group('annotations').open && tissue().visible,'Reopening keeps the folded group and visible tissue');
 const shown=s.layers.find(l=>l.type==='image');
 // Small synthetic records cover the same metadata combination as the screenshot.
 shown.info.meta=shown.channels.map((_,i)=>({dye:'AF'+(500+i),exposure_ms:150}));
 shown.channels[0].label='Pan-Cytokeratin';
 for(const roles of [true,false]){
  step(roles?'tab5':'tab4');
  await until(()=>roles ? s.markerRoles && document.querySelector('[aria-label="Cellpose model"]') : !s.markerRoles);
  for(const [dye,exposure] of [[false,false],[true,false],[false,true],[true,true]]){
   setMarkerColumn('dye',dye);setMarkerColumn('exposure',exposure);buildPanel();
   for(const width of [320,480,760]){
    document.querySelector('#app').style.setProperty('--studio-right',width+'px');await delay(30);
    const row=panel().querySelector('.channel'), visible=Array.from(row.children).filter(n=>getComputedStyle(n).display!=='none' && !n.classList.contains('resolve-link'));
    const boxes=visible.map(n=>n.getBoundingClientRect()), middle=b=>b.top+b.height/2;
    assert(boxes.every(b=>Math.abs(middle(b)-middle(boxes[0]))<2),`Marker values stay on one row: roles=${roles}, dye=${dye}, exposure=${exposure}, width=${width}`);
    assert(boxes.every((b,i)=>!i || b.left>=boxes[i-1].right-1),'Marker columns remain in order without overlapping');
    const name=row.querySelector('.cname').getBoundingClientRect(), numbers=Array.from(row.querySelectorAll('input[type=number]')).map(n=>n.getBoundingClientRect());
    assert(name.width>=109 && numbers.every(b=>b.width>=50),'Marker names and both brightness values retain readable widths');
   }
  }
 }
 document.querySelector('#app').style.setProperty('--studio-right','480px');
 step('tab5');await until(()=>s.markerRoles && document.querySelector('#studio-seg [aria-label^="Nucleus marker"]'));
 const names=shown.channels.map(c=>c.name);
 panel().querySelector('[aria-label="Nuclear marker '+names[1]+'"]').click();
 await until(()=>document.querySelector('#studio-seg [aria-label="Nucleus marker '+names[1]+'"]').checked);
 assert(true,'Aligned N buttons still choose the nuclear marker');
 panel().querySelector('[aria-label="Cell border marker '+names[0]+'"]').click();
 await until(()=>document.querySelector('#studio-seg [aria-label="Cell border marker '+names[0]+'"]').checked);
 assert(true,'Aligned M buttons still choose cell-border markers');
 document.querySelector('#studio-right [data-tab=outputs]').click();
 await until(()=>document.querySelector('.studio-output-group'));
 const output=document.querySelector('.studio-output-group');
 assert(!output.open && output.querySelectorAll('.studio-out-row').length===12,'Saved output groups start folded and keep long lists short');
 output.querySelector('summary').click();button(output,'Show all 15').click();
 assert(output.querySelectorAll('.studio-out-row').length===15,'Show all reveals every saved file in its group');
 step('tab4');await delay(300);step('tab5');
 await until(()=>document.querySelector('.studio-output-group'));
 assert(document.querySelector('.studio-output-group').open,'Returning to a step remembers its open output group');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=150, setup=setup)
