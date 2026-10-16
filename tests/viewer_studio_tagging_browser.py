"""Browser-check Studio's Cell tagging and Segmentation's "where to segment" on a disposable project in headless Firefox.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_tagging_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
fetch('/result?progress=driver%20started');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1000;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message.slice(0,80)));};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
const studioApi=path=>'/spacemap-viewer/api/studio/browser-check'+path;
const change=(element,value)=>{element.value=value;element.dispatchEvent(new Event('change'));};
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent('page error: '+event.message+' '+event.filename+':'+event.lineno)));
setTimeout(async()=>{
 try{
  const state=window.__state;
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[data-step=tab5]'));
  fetch('/result?progress=page%20ready');
  // Segmentation: one choice per image, the whole image or one of its annotations.
  document.querySelector('[data-step=tab5]').click();
  await until(()=>tools().querySelector('tr[data-image="roi_bg_subtracted.ome.tiff"]'));
  assert(tools().querySelectorAll('tr[data-image="roi_bg_subtracted.ome.tiff"] select').length===1 && !tools().querySelector('[aria-label="Where to segment"]'),
    'Each image has one choice of where to segment, and no second list');
  // Cell tagging.
  document.querySelector('[data-step=tagging]').click();
  await until(()=>tools().querySelector('.studio-tag-class[data-class="Tumor"]'));
  assert(/1 with annotations/.test(tools().textContent),'The cell table is matched to its annotation');
  const settings=async()=>(await (await fetch(studioApi('/tagging_settings'))).json()).values;
  const cpu=tools().querySelector('[aria-label="Tagging CPU cores"]');
  const compute=cpu?.closest('.studio-sec');
  const firstAnnotation=tools().querySelector('.studio-tag-class');
  assert(compute && !cpu.closest('details') && (compute.compareDocumentPosition(firstAnnotation)&Node.DOCUMENT_POSITION_FOLLOWING),
    'Tagging CPU and GPU controls are visible above the annotation list');
  assert(compute.querySelector('[aria-label="GPU distances"]') && cpu.getBoundingClientRect().height>0,
    'Compute does not need to be expanded to configure CPU or GPU');
  change(tools().querySelector('[aria-label="Tagging CPU cores"]'),'4');
  await until(async()=>(await settings()).seg_tag_cpu_workers===4);
  assert(true,'Tagging CPU workers are configurable and saved');
  tools().querySelector('[aria-label="Tumor Distance"]').click();
  await until(()=>tools().querySelector('[aria-label="Column name Tumor distance"]')?.value==='Tumor_distance');
  change(tools().querySelector('[aria-label="Column name Tumor distance"]'),'dist_tumor');
  await until(async()=>(await settings()).seg_tag_column_names['Tumor|distance']==='dist_tumor');
  assert(true,'A column is added with a tick and renamed in place');
  tools().querySelector('[aria-label="Bed Tumor"]').click();
  await delay(300);
  tools().querySelector('[aria-label="Bed Stroma"]').click();
  await until(async()=>(await settings()).seg_tag_bed_labels.length===2);
  await until(()=>tools().querySelector('[aria-label="Bed column name"]'));
  assert(true,'Beds are saved as Tab 5’s tagging settings, with one bed column');
  // Preview in memory: cells coloured by a new column, nothing written.
  const inView=tools().querySelector('[aria-label="Cells in view only"]');if(inView.checked)inView.click();
  button(tools(),'Preview this image').click();
  await until(()=>tools().querySelector('[aria-label="Colour cells by"]'));
  const cells=state.layers.find(layer=>layer.type==='cells');
  await until(()=>String(cells.color_by).startsWith('preview:'));
  change(tools().querySelector('[aria-label="Colour cells by"]'),'Tumor');
  await until(()=>cells.color_by==='preview:Tumor');
  const legend=await (await fetch('/spacemap-viewer/api/cells/'+cells.token+'/legend?t=&col=preview%3ATumor')).json();
  assert(legend.kind==='categorical' && legend.items.length===2,'The preview colours the cells by a new column, with a legend: '+JSON.stringify(legend).slice(0,200));
  change(tools().querySelector('[aria-label="Colour cells by"]'),'dist_tumor');
  await until(()=>cells.color_by==='preview:dist_tumor');
  const scale=await (await fetch('/spacemap-viewer/api/cells/'+cells.token+'/legend?t=&col=preview%3Adist_tumor')).json();
  assert(scale.kind==='numeric','A distance shows as a colour scale: '+JSON.stringify(scale).slice(0,200));
  let jobs=(await (await fetch(studioApi('/jobs'))).json()).jobs;
  assert(!jobs.some(job=>job.kind==='cell_tagging'),'Nothing is written by the preview');
  tools().querySelector('.studio-pairs th input[type=checkbox]').click();
  button(document.querySelector('.studio-runbar'),'Selected').click();
  for(let i=0;i<600;i++){jobs=(await (await fetch(studioApi('/jobs'))).json()).jobs;const job=jobs.find(item=>item.kind==='cell_tagging');if(job && /done|failed/.test(job.status))break;await delay(100);}
  const job=jobs.find(item=>item.kind==='cell_tagging');
  assert(job.status==='done' && job.result.count===1 && job.result.failed===0,'Tagging the ticked table writes it: '+job.message);
  await until(()=>tools().querySelector('tr[data-source] .chip.ok')?.textContent==='Tagged');
  assert(true,'Its row turns green');
  // Batch correction's current full-path output folder control.
  document.querySelector('[data-step=tab6]').click();
  await until(()=>tools().querySelector('[aria-label="Output folder"]'));
  change(tools().querySelector('[aria-label="Output folder"]'),'outputs/combined_check');
  await until(async()=>(await (await fetch(studioApi('/folders/tab6'))).json()).folders.some(row=>row.folder==='outputs/combined_check'));
  await until(()=>tools().querySelector('[aria-label="Output folder"]')?.value.endsWith('/outputs/combined_check'));
  assert(true,'A step’s output folder is changed in the step, saved as the project’s folder');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(-900)));}
},100);
</script>
"""


def _polygon(name, x0, y0, x1, y1):
    return {"type": "Feature", "properties": {"classification": {"name": name}},
            "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


def setup(root):
    """The ROI's own cell table and annotation (tissue; Tumor left, Stroma right). The background-removed copy of the
    ROI has no annotation of its own: its match is picked by hand."""
    import json
    import shutil

    import pandas as pd
    from spacemap_gui.studio.correction import save_folders

    # The shared viewer fixture also has a background-subtracted copy of this
    # ROI. Batch correction needs one image per ROI, unlike the viewer.
    images = root / "correction_images"
    images.mkdir()
    shutil.copyfile(root / "roi.ome.tiff", images / "roi.ome.tiff")
    save_folders(root, {"images": str(images)})

    features = root / "outputs/segmentation/cellpose/features"
    features.mkdir(parents=True, exist_ok=True)
    pd.read_csv(root / "cells.csv").to_csv(features / "roi_cell_features.csv", index=False)
    notes = root / "outputs/processed_annotations/stage1b_compartments_zones"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "roi_zones.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        _polygon("tissue", 0, 0, 400, 400), _polygon("Tumor", 0, 0, 160, 400), _polygon("Stroma", 160, 0, 400, 400)]}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
