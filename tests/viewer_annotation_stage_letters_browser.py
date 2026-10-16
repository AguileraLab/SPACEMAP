"""Browser-check the Annotation list's stage letters and the Artifacts batch row: a click on a letter opens its image
at that stage with the stage's polygons shown; Artifacts finds or subtracts for the ticked images as one job.

Usage: PYTHONPATH=src:tests python tests/viewer_annotation_stage_letters_browser.py
"""
from viewer_threshold_browser import main
from viewer_measure_regions_browser import setup as tissue_setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent.startsWith(t));
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools'),bar=()=>document.querySelector('.studio-runbar');
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 const letter=id=>tools().querySelector('tr[data-image="markers.ome.tiff"] [data-annotation-stage='+id+']');
 await until(()=>letter('artifacts')?.classList.contains('done'));
 const activeStage=()=>tools().querySelector('.studio-stage-tabs .active')?.textContent;
 const layer=id=>s.layers.find(l=>l.id===id);
 letter('artifacts').click();
 await until(()=>activeStage()==='Artifacts'&&layer('artifact-clot')?.visible&&layer('artifact-clot').features?.length);
 assert(s.layers.find(l=>l.type==='image')?.name==='markers.ome.tiff','A opens its image');
 assert(layer('artifact-clot').visible&&layer('drawn-artifacts').visible,'A shows the artifact polygons (Blood clots and the others)');
 await until(()=>bar()&&button(bar(),'Selected'));
 assert(/Find:/.test(bar().textContent)&&/Replace saved/.test(bar().textContent),'Artifacts has Find: Replace saved, This image, Selected');
 layer('tissue-regions').visible=false;
 letter('tissue').click();
 await until(()=>activeStage()==='Tissue'&&layer('tissue-regions')?.visible);
 assert(true,'T switches to Tissue and shows the tissue');
 letter('regions').click();
 await until(()=>activeStage()==='Marker regions');
 assert(true,'A grey letter goes to its stage too');
 // Subtract for the ticked images: one job, the tissue smaller after it.
 letter('artifacts').click();
 await until(()=>activeStage()==='Artifacts'&&bar()&&Array.from(bar().querySelectorAll('button')).filter(b=>b.textContent==='Selected').length===2);
 const subtractSome=Array.from(bar().querySelectorAll('button')).filter(b=>b.textContent==='Selected')[1];
 subtractSome.click();
 await until(()=>/Tick the images first/.test(bar().textContent));
 assert(true,'Selected asks for ticked images first');
 tools().querySelector('[aria-label="Pick markers.ome.tiff"]').click();
 subtractSome.click();
 await until(()=>/Artifacts · ticked images/.test(document.body.textContent));
 await until(async()=>{const r=await fetch('/spacemap-viewer/api/studio/browser-check/jobs').then(r=>r.json()).catch(()=>null);return (r?.jobs||[]).some(j=>j.kind==='artifact_batch'&&j.status==='done');});
 assert(true,'The subtract job finishes');
 const jobList=async()=>((await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs||[]);
 // Both kinds at once, for the open image.
 bar().querySelector('[aria-label="Blood clots and Image quality together"]').click();
 button(bar(),'Subtract from tissue').click();
 await until(async()=>(await jobList()).filter(j=>j.kind==='artifact_batch'&&j.status==='done').length===2);
 assert(true,'Both kinds subtracts Blood clots and Image quality for the open image');
 // Remove small pieces: this image at once, then the ticked ones as a job.
 letter('tissue').click();
 await until(()=>activeStage()==='Tissue'&&bar()?.querySelector('[aria-label="Remove pieces smaller than (µm²)"]'));
 const small=bar().querySelector('[aria-label="Remove pieces smaller than (µm²)"]');small.value='1';
 const smallRow=small.closest('.row');button(smallRow,'This image').click();
 await until(()=>/No piece is smaller|removed/.test(bar().textContent));
 assert(true,'Remove small pieces: This image works on the open tissue');
 small.value='1';button(smallRow,'Selected').click();
 await until(async()=>(await jobList()).some(j=>j.kind==='remove_small_pieces'&&j.status==='done'));
 assert(true,'Remove small pieces: Selected runs on the ticked images');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,800)));}},100);
</script>
"""


def setup(root):
    """The marker image, its tissue and a saved Blood clots polygon inside it."""
    import json

    from shapely.geometry import box, mapping

    from spacemap_gui.studio import artifact_learning

    tissue_setup(root)
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    path = artifact_learning.polygons_path(output, "markers", "clot")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {
        "classification": {"name": artifact_learning.CLASSES["clot"]}}, "geometry": mapping(box(40, 40, 100, 120))}]}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180, setup=setup)
