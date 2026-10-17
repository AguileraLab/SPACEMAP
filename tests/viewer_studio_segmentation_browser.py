"""Browser-check Studio's Segmentation step on a disposable image in headless Firefox.

Runs Cellpose (the cached cyto and nuclei models) on the CPU unless the node has a GPU.
Usage: PYTHONPATH=src:tests python tests/viewer_studio_segmentation_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<4000;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab5]'));
  document.querySelector('[data-step=tab5]').click();
  await until(()=>document.querySelector('#studio-seg [aria-label="Cellpose model"]') && document.querySelector('#studio-tools tr[data-image="roi.ome.tiff"]'));
  const viewer=state.viewer, form=()=>document.querySelector('#studio-seg'), tools=()=>document.querySelector('#studio-tools');
  const settings=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/segmentation_settings')).json()).values;
  const image=state.layers.find(item=>item.type==='image');
  const names=image.channels.map(channel=>channel.name);
  const nucleus=name=>form().querySelector('[aria-label="Nucleus marker '+name+'"]'), border=name=>form().querySelector('[aria-label="Cell border marker '+name+'"]');
  assert(names.every(name=>nucleus(name) && border(name)) && names.filter(name=>nucleus(name).checked).length===1 && !names.some(name=>border(name).checked),
    'Every marker has a Nucleus and a Border choice; one nucleus, no border to start');
  const first=names.find(name=>nucleus(name).checked);
  // N and M on the image's markers in Layers set the same roles; settings save by themselves.
  await until(()=>document.querySelector('#panel [aria-label="Nuclear marker '+names[1]+'"]'));
  assert(document.querySelector('#panel [aria-label="Nuclear marker '+first+'"]').getAttribute('aria-pressed')==='true','The nuclear marker is lit N in Layers');
  document.querySelector('#panel [aria-label="Nuclear marker '+names[1]+'"]').click();
  await until(()=>nucleus(names[1]).checked);
  document.querySelector('#panel [aria-label="Cell border marker '+names[0]+'"]').click();
  await until(()=>border(names[0]).checked);
  await until(async()=>{const v=await settings();return v.tab4_last_nuclear_names[0]===names[1]&&v.tab4_last_membrane_names.includes(names[0]);});
  assert(document.querySelector('#panel [aria-label="Cell border marker '+names[0]+'"]').getAttribute('aria-pressed')==='true','N and M in Layers set the roles, saved without a Save button');
  document.querySelector('#panel [aria-label="Cell border marker '+names[0]+'"]').click();
  document.querySelector('#panel [aria-label="Nuclear marker '+first+'"]').click();
  await until(()=>nucleus(first).checked && !border(names[0]).checked);
  // One model list, grouped as the artifact models are.
  const model=form().querySelector('[aria-label="Cellpose model"]');
  assert(Array.from(model.querySelectorAll('optgroup')).map(g=>g.label).includes('Cellpose'),'Models are one list, in groups');
  model.value='nuclei';model.dispatchEvent(new Event('change'));
  await until(()=>/is the model now/.test(tools().textContent));
  assert((await settings()).tab4_model_type==='nuclei','Choosing a model saves it as Tab 5\'s model');
  const test=button(tools(),'Test on this view');test.click();
  await until(()=>!test.disabled && /cells ·|error|Could/.test(test.parentElement.parentElement.textContent));
  const text=test.parentElement.parentElement.textContent;
  assert(/\d+ cells · [\d.]+ s · (GPU|CPU)/.test(text),'The test reports cells, time and device: '+text);
  assert(state.segPreview && document.querySelector('#shapes path.seg-preview-box'),'The tested box and its cells are drawn on the image');
  const bar=document.querySelector('.studio-runbar');
  button(bar,'Selected').click();
  await until(()=>/Tick the images first/.test(bar.textContent));
  button(bar,'This image').click();
  await until(()=>/cells in 1 image|failed/.test(document.querySelector('#studio-jobs').textContent));
  const jobs=await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json();
  assert(jobs.jobs[0].kind==='segment_images' && jobs.jobs[0].status==='done','The image is segmented by a job: '+jobs.jobs[0].message);
  const row=()=>tools().querySelector('tr[data-image="roi.ome.tiff"]');
  await until(()=>/[\d,]+ cells/.test(row()?.querySelector('.chip.ok')?.textContent||''));
  assert(row().querySelector('.chip.ok'),'Its row turns green with the number of cells');
  assert(state.viewer===viewer,'The image stays open through testing and segmenting');
  // An image segmented before (two cells saved): opening it shows its cells in Layers.
  const done=tools().querySelector('tr[data-image="roi_bg_subtracted.ome.tiff"]');
  assert(/^2 cells$/.test(done.querySelector('.chip.ok')?.textContent||''),'Images segmented earlier (by classic too) are green with their cells counted');
  done.querySelector('button.studio-item-name').click();
  await until(()=>state.layers.some(layer=>layer.type==='cells' && layer.visible && /cell_boundaries/.test(layer.source)));
  assert(/2 cells, shown on the image/.test(tools().textContent),'Opening it shows its cells in Layers and on the image');
  document.querySelector('[data-step=tab4]').click();
  await until(()=>!document.querySelector('#panel .role-toggle'));
  assert(true,'Leaving Segmentation takes N and M away');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,test:text})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}
},100);
</script>
"""


def setup(root):
    """Results of an earlier segmentation of the background-removed ROI: two numbered cells and their table."""
    import json

    output = root / "outputs/segmentation"
    (output / "geojson").mkdir(parents=True, exist_ok=True)
    cells = [{"type": "Feature", "properties": {"label": n}, "geometry": {"type": "Polygon", "coordinates": [[[x, 20], [x + 15, 20],
              [x + 15, 35], [x, 35], [x, 20]]]}} for n, x in ((1, 20), (2, 60))]
    (output / "geojson/roi_bg_subtracted.ome_cell_boundaries.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": cells}))
    (output / "roi_bg_subtracted.ome_cell_features.csv").write_text("label,x_centroid,y_centroid\n1,27,27\n2,67,27\n")


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
