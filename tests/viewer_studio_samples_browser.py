"""Browser-check Studio's Samples (Batch correction step) on a disposable project in headless Firefox.

Slide S1 has two ROIs in its ROI file; S2_roi_1 has a cell table only, so its slide is read from its name.
The check corrects that slide by hand, adds a clinic table linked by slide, renames and removes a column.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_samples_browser.py
"""

import json

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab6]'));
  document.querySelector('[data-step=tab6]').click();
  const samplesDetails=Array.from(document.querySelectorAll('#studio-tools .studio-sec > details')).find(d=>d.querySelector('summary').textContent.startsWith('Samples'));
  assert(samplesDetails && !samplesDetails.open, 'Samples starts folded'); samplesDetails.open=true;
  const box=()=>document.querySelector('.studio-samples')?.closest('.studio-sec');
  await until(()=>box() && box().querySelector('[aria-label="slide of S2_roi_1"]'));
  const text=()=>box().textContent;
  const slideOf=roi=>box().querySelector('[aria-label="slide of '+roi+'"]');
  assert(slideOf('S1_roi_1').value==='S1' && slideOf('S1_roi_2').value==='S1' && !slideOf('S1_roi_1').classList.contains('guess') && /slides? from names only/.test(text()) && !/null|undefined/.test(text()),'Samples lists the ROIs with the slide the ROI step recorded (no stray text): '+text().slice(0,200));
  const slide=box().querySelector('[aria-label="slide of S2_roi_1"]');
  assert(slide.value==='S2' && slide.classList.contains('guess'),'A slide read from the name is marked');
  slide.value='S1';slide.dispatchEvent(new Event('change'));
  await until(()=>/S2_roi_1: slide saved/.test(text()));
  let saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/samples')).json();
  assert(saved.saved && saved.rows.find(r=>r.roi==='S2_roi_1').slide==='S1','A slide corrected by hand is saved');
  // A table from the clinic: one row per slide.
  const input=box().querySelector('input[type=file]');
  const transfer=new DataTransfer();transfer.items.add(new File(['Slide name,Outcome,Arm\nS1,CR,A\nS2,PD,B\n'],'clinic.csv',{type:'text/csv'}));
  input.files=transfer.files;input.dispatchEvent(new Event('change'));
  await until(()=>box().querySelector('[aria-label="Link column"]'));
  assert(box().querySelector('[aria-label="Link column"]').value==='Slide name' && box().querySelector('[aria-label="Match in dataset"]').value==='slide','The link column and level are suggested');
  const outcome=box().querySelector('[aria-label="Name for Outcome"]');outcome.value='response';
  button(box(),'Preview').click();
  await until(()=>/3 of \d+ ROIs matched/.test(text()));
  assert(/not used: S2/.test(text()),'The preview says what matched and what was not used');
  button(box(),'Add these columns').click();
  await until(()=>Array.from(box().querySelectorAll('th')).map(th=>th.textContent).join()==='ROI,Slide,response,Arm');
  assert(box().querySelector('[aria-label="response of S1_roi_2"]').value==='CR','The columns are added to every ROI of the slide');
  await until(()=>document.querySelector('[aria-label="Keep biological differences in response"]'));
  assert(Array.from(document.querySelector('[aria-label="Correct between"]').options).some(o=>o.value==='response'), 'Imported clinical columns immediately appear in correction settings');
  box().querySelector('[aria-label="Rename Arm"]').click();
  const rename=box().querySelector('[aria-label="New name for Arm"]');rename.value='arm';
  button(box(),'Save').click();
  await until(()=>Array.from(box().querySelectorAll('th')).some(th=>th.textContent==='arm'));
  box().querySelector('[aria-label="Remove arm"]').click();
  await until(()=>Array.from(box().querySelectorAll('th')).map(th=>th.textContent).join()==='ROI,Slide,response');
  assert(true,'Columns can be renamed and removed');
  button(box(),'Add column').click(); box().querySelector('[aria-label="New sample column"]').value='Stage'; button(box(),'Add').click();
  await until(()=>box().querySelector('[aria-label="Stage of S1_roi_1"]'));
  const stage=box().querySelector('[aria-label="Stage of S1_roi_1"]'); stage.value='II'; stage.dispatchEvent(new Event('change'));
  await until(()=>/Stage saved/.test(text()));
  assert((await (await fetch('/spacemap-viewer/api/studio/browser-check/samples')).json()).rows.find(r=>r.roi==='S1_roi_1').Stage==='II', 'Clinical data can also be entered by hand');
  const clinic=new DataTransfer(); clinic.items.add(new File(['Case,Clinical\nP1,Positive\nP2,Negative\n'],'patient.csv',{type:'text/csv'}));
  input.files=clinic.files; input.dispatchEvent(new Event('change'));
  await until(()=>box().querySelector('[aria-label="Match in dataset"]'));
  const target=box().querySelector('[aria-label="Match in dataset"]');
  assert(Array.from(target.options).some(o=>o.value==='column:PatientID'), 'Import offers columns inside the matched cell tables');
  target.value='column:PatientID'; target.dispatchEvent(new Event('change'));
  button(box(),'Preview').click(); await until(()=>/3 of 3 ROIs matched/.test(text()));
  button(box(),'Add these columns').click(); await until(()=>box().querySelector('[aria-label="Clinical of S2_roi_1"]'));
  assert(box().querySelector('[aria-label="Clinical of S1_roi_2"]').value==='Positive' && box().querySelector('[aria-label="Clinical of S2_roi_1"]').value==='Negative', 'Dataset matching uses PatientID even when Slide IDs have been edited');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,600)));}
},100);
</script>
"""


def setup(root):
    ring = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    boxes = [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"name": name}}
        for name in ("roi_1", "roi_2")
    ]
    shapes = root / "data/boundaries"
    shapes.mkdir(parents=True, exist_ok=True)
    (shapes / "S1_roi_bounding_boxes_coords.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": boxes})
    )
    tables = root / "outputs/segmentation"
    tables.mkdir(parents=True, exist_ok=True)
    images = root / 'batch_rois'
    images.mkdir()
    config_path = root / (root.name + '.json')
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.setdefault('directory_mappings', {})['bc_images_dir'] = str(images)
    config_path.write_text(json.dumps(config))
    for roi in ("S1_roi_1", "S1_roi_2", "S2_roi_1"):
        (images / (roi + '.ome.tiff')).touch()
        patient = 'P1' if roi.startswith('S1') else 'P2'
        (tables / f"{roi}.ome_cell_features_tagged.csv").write_text(f"label,M0,PatientID\n1,5,{patient}\n")


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=240, setup=setup)
