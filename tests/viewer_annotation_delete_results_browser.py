"""Browser-check Delete results in the Annotation step.

Export the open image (its E letter turns green), then Delete results… > This image on the Export stage: the letter
turns grey again and the tissue stays.

Usage: PYTHONPATH=src:tests python tests/viewer_annotation_delete_results_browser.py
"""

from viewer_measure_regions_browser import setup
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0; const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
const studioApi=path=>'/spacemap-viewer/api/studio/browser-check'+path;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent('page error: '+e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try {
 const s=window.__state, tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.layers.find(l=>l.id==='tissue-regions')?.features?.length===3 && s.imagesReady);
 const letter=id=>tools().querySelector('tr[data-image="markers.ome.tiff"] [data-annotation-stage="'+id+'"]');
 button(document.querySelector('.studio-stage-tabs'),'Export').click();
 await until(()=>tools().querySelector('[aria-label="Export tissue (local)"]'));
 tools().querySelector('[aria-label="Export tissue (local)"]').click();
 await until(async()=>(await (await fetch(studioApi('/annotation_export'))).json()).settings.rules['local:tissue']?.export===true);
 assert(true,'Tissue is ticked for export');
 // Export, then delete the export.
 const click=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(!this.download)click.call(this);};
 button(document.querySelector('.studio-runbar'),'This image').click();
 await until(()=>/annotations exported/.test(document.querySelector('.studio-runbar').textContent));
 await until(()=>letter('export')?.classList.contains('done'));
 assert(true,'Export turns E green');
 window.confirm=()=>true;
 button(tools(),'Delete results…').click();
 const row=()=>tools().querySelector('.studio-delete-results');
 await until(()=>row()&&!row().hidden&&/Delete Export results of/.test(row().textContent));
 button(row(),'This image').click();
 await until(()=>/set aside in outputs\/studio\/deleted/.test(row().textContent));
 await until(()=>letter('export')&&!letter('export').classList.contains('done'));
 assert(letter('tissue').classList.contains('done'),'Delete results turns E grey and leaves the tissue');
 HTMLAnchorElement.prototype.click=click;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(document.querySelector('#studio-tools')?.textContent||'').slice(0,900)));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300, setup=setup)
