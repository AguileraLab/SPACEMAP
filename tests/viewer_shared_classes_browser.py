"""Browser-check shared classes: found on two ticked images, chosen, named once on one image and shown on the other,
saved as polygons for both; each image its own classes still works.

Usage: PYTHONPATH=src:tests python tests/viewer_shared_classes_browser.py
"""
from viewer_threshold_browser import main
from viewer_artifact_review_browser import setup as artifact_setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<2400;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools'),footer=()=>document.querySelector('.studio-runbar');
 const stage=name=>button(document.querySelector('.studio-stage-tabs'),name).click();
 const barRow=label=>Array.from(footer().querySelectorAll('.row')).find(r=>r.querySelector('.lbl')?.textContent===label);
 const jobs=async()=>((await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs||[]);
 const open=async name=>{tools().querySelector('tr[data-image="'+name+'"] button.studio-item-name').click();
   await until(()=>s.imagesReady&&!s.commandBusy&&s.layers.find(l=>l.type==='image')?.name===name);};
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="mirror.ome.tiff"] button.studio-item-name'));
 await open('markers.ome.tiff');
 stage('Classification');
 await until(()=>tools().querySelector('[aria-label="Class model"]')&&/Not classified in this area yet/.test(tools().textContent));
 assert(tools().querySelector('[aria-label="Class model"]').value===''&&barRow('Shared classes:'),'Each image its own classes by default; a Shared classes row');
 const count=tools().querySelector('[aria-label="Number of classes"]');count.value='3';count.dispatchEvent(new Event('change'));
 for(const name of ['markers.ome.tiff','mirror.ome.tiff'])tools().querySelector('[aria-label="Pick '+name+'"]').click();
 button(barRow('Shared classes:'),'Selected').click();
 await until(async()=>(await jobs()).some(j=>j.kind==='classify_shared'&&j.status==='done'));
 const job=(await jobs()).find(j=>j.kind==='classify_shared');
 assert(job.result.fitted&&job.result.count===2&&job.result.n_clusters===3,'One set of classes is found on both images and each is classified with it');
 await until(()=>tools().querySelector('[aria-label="Class model"]').value===job.result.model&&tools().querySelector('[aria-label="Number of classes"]').disabled);
 assert(true,'The new model is chosen; its settings are shown and fixed');
 await until(()=>/Shared classes:/.test(tools().querySelector('.classification-list')?.textContent||'')&&tools().querySelectorAll('.classification-row').length);
 const rows=Array.from(tools().querySelectorAll('.classification-row'));
 const cd3=rows.find(r=>/cd3/i.test(r.querySelector('.classification-markers').textContent))||rows[0];
 const id=cd3.dataset.class,nameBox=cd3.querySelector('input[type=text]');
 nameBox.value='T cell zone';button(cd3,'✓').click();
 await until(()=>tools().querySelector('.classification-row[data-class="'+id+'"] input[type=text]')?.value==='T cell zone'&&!button(barRow('Classify:'),'This image').disabled);
 const markersRun=document.querySelector('.classification-list').textContent;
 await open('mirror.ome.tiff');
 await until(()=>tools().querySelector('.classification-row[data-class="'+id+'"]')&&/fit no class|Shared classes:/.test(tools().querySelector('.classification-list').textContent)
   &&/pixels · saved run/.test(tools().textContent)&&!button(barRow('Classify:'),'This image').disabled);
 assert(tools().querySelector('.classification-row[data-class="'+id+'"] input[type=text]').value==='T cell zone','A name given on one image is the class name on the other');
 await until(()=>!button(barRow('Save polygons:'),'Selected').disabled);
 button(barRow('Save polygons:'),'Selected').click();
 await until(async()=>(await jobs()).some(j=>j.kind==='save_class_polygons'&&j.status==='done')).catch(async()=>{
   throw Error('Save batch: '+footer().textContent.slice(-300)+' | jobs '+JSON.stringify((await jobs()).map(j=>[j.kind,j.status,j.message])));});
 const saved=(await jobs()).find(j=>j.kind==='save_class_polygons');
 assert(saved.result.count===2,'Save polygons: Selected writes both images');
 // This image with the model, then each image its own again.
 button(barRow('Classify:'),'This image').click();
 await until(()=>/Shared classes:/.test(tools().querySelector('.classification-list')?.textContent||'')&&!button(barRow('Classify:'),'This image').disabled);
 const picker=tools().querySelector('[aria-label="Class model"]');picker.value='';picker.dispatchEvent(new Event('change'));
 await until(()=>!tools().querySelector('[aria-label="Number of classes"]').disabled);
 button(barRow('Classify:'),'This image').click();
 await until(()=>tools().querySelectorAll('.classification-row').length&&!/Shared classes:/.test(tools().querySelector('.classification-list').textContent)&&!button(barRow('Classify:'),'This image').disabled);
 assert(true,'Each image its own classes still classifies the open image as before');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,1200)));}},100);
</script>
"""


def setup(root):
    """Two images with the same tissues mirrored, each with its tissue."""
    import json

    import numpy as np
    import tifffile
    from shapely.geometry import box, mapping

    from spacemap_gui.studio import annotation
    from test_studio_artifacts import marker_image

    artifact_setup(root)
    data = tifffile.imread(root / "markers.ome.tiff")
    marker_image(root / "mirror.ome.tiff", np.ascontiguousarray(data[:, :, ::-1]), ["HE_Red", "HE_Green", "HE_Blue", "CD3", "PanCK"])
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    annotation.valid_tissue(output, "mirror").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(box(10, 10, 390, 390))}]}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300, setup=setup)
