"""Browser-check the Annotation step's Sub-directory: an existing project moves its files only when asked, and its
image opens with its tissue from the new place.

Usage: PYTHONPATH=src:tests python tests/viewer_annotation_subfolder_browser.py
"""
from viewer_threshold_browser import main
from viewer_measure_regions_browser import setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent.startsWith(t));
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('[aria-label="Sub-directory"]'));
 const box=tools().querySelector('[aria-label="Sub-directory"]');
 assert(!box.checked&&button(tools(),'Move '),'An existing project keeps its files where they are and offers to move them');
 button(tools(),'Move ').click();
 await until(()=>/moved into .*working_files/.test(tools().textContent));
 assert(tools().querySelector('[aria-label="Sub-directory"]').checked&&!button(tools(),'Move '),'Move turns the Sub-directory on');
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 await until(()=>s.imagesReady&&tissue()?.features?.length);
 assert(/working_files/.test(tissue().saved_file),'The image opens with its tissue from the sub-directory: '+tissue().saved_file);
 assert(tools().querySelector('tr[data-image="markers.ome.tiff"] [data-annotation-stage=tissue].done'),'The list still shows its tissue');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,800)));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=120, setup=setup)
