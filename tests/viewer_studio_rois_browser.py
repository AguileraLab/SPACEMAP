"""Browser-check Studio's ROI step on a disposable image in headless Firefox.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_rois_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<2200;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab3]'));
  document.querySelector('[data-step=tab3]').click();
  await until(()=>button(document.querySelector('#studio-rois'),'Show its ROIs'));
  const image=state.layers.find(item=>item.type==='image'), viewer=state.viewer;
  button(document.querySelector('#studio-rois'),'Show its ROIs').click();
  await until(()=>state.layers.some(item=>item.id==='rois') && document.querySelector('#studio-rois .shape-list'));
  const closeLayer=button(document.querySelector('#studio-action-status'),'Undo');
  assert(closeLayer,'Opening the ROI layer can be undone');
  const rois=state.layers.find(item=>item.id==='rois');
  assert(rois.editable,'The ROI layer is open for drawing');
  assert(!document.querySelector('#drawtools').hidden,'Drawing tools appear above the image');
  document.querySelector('#drawtools [data-tool=rect]').click();
  const position=(x,y)=>viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(x,y));
  viewer.raiseEvent('canvas-press',{position:position(20,20)});
  viewer.raiseEvent('canvas-drag',{position:position(120,90)});
  viewer.raiseEvent('canvas-release',{position:position(120,90)});
  await until(()=>(rois.features||[]).length===1 && document.querySelector('#studio-rois .shape-row input[title=Name]'));
  const name=document.querySelector('#studio-rois .shape-row input[title=Name]');
  name.value='tumour';name.dispatchEvent(new Event('change'));
  await until(()=>rois.features[0].properties.name==='tumour' && (rois.status||'').startsWith('Saved'));
  assert(true,'A drawn ROI is named and saved into the ROI file');
  const bar=document.querySelector('.studio-runbar');
  const row=text=>Array.from(bar.querySelectorAll('.row')).find(item=>item.querySelector('.lbl')?.textContent===text);
  button(row('Extract'),'This image').click();
  await until(()=>document.querySelector('#studio-jobs').textContent.includes('ROI image'));
  const jobs=await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json();
  assert(jobs.jobs[0].kind==='extract_rois' && jobs.jobs[0].status==='done','The job cuts the ROI out with Tab 3');
  assert(jobs.jobs[0].result.images.some(path=>path.endsWith('roi_tumour.ome.tiff')),'The ROI name becomes the image name');
  closeLayer.click();
  await until(()=>!state.layers.some(item=>item.id==='rois') && button(document.querySelector('#studio-rois'),'Show its ROIs'));
  button(document.querySelector('#studio-rois'),'Show its ROIs').click();
  await until(()=>state.layers.some(item=>item.id==='rois') && document.querySelector('#studio-rois .shape-row input[title=Name]'));
  assert(state.layers.find(item=>item.id==='rois').features[0].properties.name==='tumour','Reopening loads the saved ROIs from the file');
  assert(state.viewer===viewer,'The image stays open throughout');
  // Markers: keep, colour, the name in its box, dye and exposure; saved for this image as they change.
  const panel=document.querySelector('#studio-tools');
  await until(()=>panel.querySelector('[aria-label="Extract M1"]'));
  assert(panel.querySelector('[aria-label="Extract M0"]').checked && /2 of 2 markers/.test(panel.textContent),'Every marker is extracted until chosen otherwise');
  assert(panel.querySelector('[aria-label="Name of M0"]').value==='M0' && panel.querySelector('.studio-roi-marker .studio-meta'),'The name is the box; dye and exposure are shown');
  panel.querySelector('[aria-label="Extract M1"]').click();
  const rename=panel.querySelector('[aria-label="Name of M0"]');rename.value='DNA';rename.dispatchEvent(new Event('change'));
  await until(()=>/Markers saved for/.test(panel.textContent));
  const markers=await (await fetch('/spacemap-viewer/api/studio/browser-check/roi_markers')).json();
  assert(JSON.stringify(markers.markers.map(m=>[m.name,m.keep,m.label]))==='[["M0",true,"DNA"],["M1",false,"M1"]]','The choice is saved for this image: '+JSON.stringify(markers.markers));
  // The list: a TMA image in the images folder; select all; open it with its ROIs; Detect cores; Extract the ticked.
  await until(()=>panel.querySelector('tr[data-image="T1_merged.ome.tiff"]'));
  assert(!button(panel,'Pick all') && !panel.querySelector('.studio-work'),'No Pick all button and no second list');
  panel.querySelector('.studio-pairs th input[type=checkbox]').click();
  assert(panel.querySelector('tr[data-image="T1_merged.ome.tiff"] td input[type=checkbox]').checked,'The box on top ticks every image');
  panel.querySelector('tr[data-image="T1_merged.ome.tiff"] button.studio-item-name').click();
  await until(()=>/T1_merged.ome.tiff with its ROIs/.test(panel.textContent) && state.layers.some(item=>item.id==='rois') && panel.querySelector('[aria-label="Name of DAPI"]'));
  assert(/No ROIs/.test(panel.querySelector('tr[data-image="T1_merged.ome.tiff"]').textContent),'The row shows it has no ROIs yet');
  button(row('Detect cores'),'This image').click();
  await until(()=>/6 cores found on T1_merged/.test(bar.textContent));
  await until(()=>(state.layers.find(item=>item.id==='rois').features||[]).length===6);
  assert(state.layers.find(item=>item.id==='rois').features.map(f=>f.properties.name).join()==='A1,A2,A3,B1,B2,B3','The cores are the image’s ROIs, named by row and column');
  await until(()=>/6 ROIs/.test(panel.querySelector('tr[data-image="T1_merged.ome.tiff"]').textContent));
  const whole=row('Extract').querySelector('[aria-label="Keep the whole box around each ROI"]');
  assert(whole && !whole.checked,'Extract blanks outside each shape unless Whole box is ticked');
  button(row('Extract'),'Selected').click();
  let done=null;
  for(let i=0;i<900 && !done;i++){const list=(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
    done=list.find(job=>job.kind==='extract_rois' && /done|failed/.test(job.status) && job.result && job.result.images.some(path=>/T1_A1\.ome\.tiff$/.test(path)));
    if(!done)await delay(200);}
  assert(done && done.status==='done' && done.result.images.length===6,'Extract cuts out every core of the ticked image');
  await until(()=>panel.querySelector('tr[data-image="T1_merged.ome.tiff"] .chip.ok')?.textContent==='6 extracted');
  assert(true,'The list follows the job: the row turns green without reloading');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}
},100);
</script>
"""


def setup(root):
    """A small tissue microarray in the registered folder: a 2 x 3 grid of round cores."""
    import cv2
    import numpy as np
    import tifffile

    (root / "outputs/registered").mkdir(parents=True, exist_ok=True)
    dapi = np.zeros((700, 1050), np.uint16)
    for row in range(2):
        for column in range(3):
            cv2.circle(dapi, (175 + column * 350, 175 + row * 350), 90, 4000, -1)
    tifffile.imwrite(root / "outputs/registered/T1_merged.ome.tiff", np.stack([dapi, dapi // 2]), metadata={
        "axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "Channel": {"Name": ["DAPI", "CD3"]}})


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300, setup=setup)
