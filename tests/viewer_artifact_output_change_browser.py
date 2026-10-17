"""Browser-check that Artifacts saves into the current Output folder after it is changed in the same page (the
layer remembered from before pointed at the old folder).

Usage: PYTHONPATH=src:tests python tests/viewer_artifact_output_change_browser.py
"""
from viewer_threshold_browser import main
from viewer_artifact_review_browser import setup as artifact_setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1000;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 const stage=name=>button(document.querySelector('.studio-stage-tabs'),name).click();
 const open=async name=>{tools().querySelector('tr[data-image="'+name+'"] button.studio-item-name').click();
   await until(()=>s.imagesReady&&!s.commandBusy&&s.layers.find(l=>l.type==='image')?.name===name);};
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 await open('markers.ome.tiff');
 stage('Artifacts');
 const clot=()=>s.layers.find(l=>l.id==='artifact-clot');
 await until(()=>clot()?.saved_file&&tools().querySelector('[data-artifact-training=clot]'));
 assert(/stage0_tissue_annotation/.test(clot().saved_file),'At first the Blood clots layer is in the Output folder');
 const output=tools().querySelector('[aria-label="Output folder"]');
 output.value=output.value.replace('stage0_tissue_annotation','other_output');output.dispatchEvent(new Event('change'));
 await until(()=>/Folder changed/.test(tools().textContent));
 await open('roi.ome.tiff');await open('markers.ome.tiff');
 stage('Artifacts');
 await until(()=>clot()?.saved_file&&/other_output/.test(clot().saved_file));
 assert(true,'After the Output folder changes, the Blood clots layer is the new folder’s ('+clot().saved_file.split('/').slice(-4).join('/')+')');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+(window.__state.layers.find(l=>l.id==='artifact-clot')||{}).saved_file));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180, setup=artifact_setup)
