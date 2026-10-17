"""Browser-check Studio's batch correction run on a disposable project in headless Firefox.

Slide S1 (two ROIs) and S2 (one ROI, stained brighter): the step suggests correcting between slides with
cyCombine's method, runs Tab 6's steps as a job, saves the choices in Tab 6's settings, and the new table
becomes Studio's combined table, measured between slides.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_correction_browser.py
"""

import json

import numpy as np
import pandas as pd
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<3600;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab6]'));
  document.querySelector('[data-step=tab6]').click();
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Correct between"]'));
  const unit=tools().querySelector('[aria-label="Correct between"]'), method=tools().querySelector('[aria-label="Method"]');
  assert(unit.value==='slide' && unit.selectedOptions[0].textContent==='slide (2)','The slide is suggested as the unit: '+unit.selectedOptions[0].textContent);
  assert(method.value==='cycombine' && method.selectedOptions[0].textContent==='cyCombine','cyCombine’s method is suggested when the project never chose one');
  const somSteps=()=>tools().querySelector('[aria-label="SOM iterations"]').closest('.field');
  assert(tools().querySelector('.studio-step-body > .studio-sec .studio-sec-h').textContent.includes('Folders and tables'), 'Folders and tables are first');
  const inputFolder=tools().querySelector('[aria-label="Cell tables folder"]'), outputFolder=tools().querySelector('[aria-label="Output folder"]');
  assert(inputFolder?.value.startsWith('/') && outputFolder?.value.startsWith('/'), 'Full directory paths at the top');
  assert(tools().querySelector('[aria-label="Images folder"]')?.value.startsWith('/'), 'ROI images have their own directory');
  assert(inputFolder.compareDocumentPosition(tools().querySelector('tr[data-roi]')) & Node.DOCUMENT_POSITION_FOLLOWING, 'Directories precede the selectable table list');
  assert(somSteps().hidden && !!tools().querySelector('[data-tip^="Python port of cyCombine"]'),'cyCombine hides the settings it does not use and says what it does');
  assert(tools().querySelector('[aria-label="Correct M0"]').checked && tools().querySelector('[aria-label="Correct M1"]').checked && !tools().querySelector('[aria-label="Correct area"]').checked,'The markers are picked, not the cell size');
  method.value='monotonic';method.dispatchEvent(new Event('change'));
  assert(!somSteps().hidden && !!tools().querySelector('[data-tip*="no cell changes place"]'),'Monotonic shows its own settings and what it keeps');
  const steps=tools().querySelector('[aria-label="SOM iterations"]');steps.value='2000';steps.dispatchEvent(new Event('change'));
  method.value='cycombine';method.dispatchEvent(new Event('change'));
  const rows=()=>Array.from(tools().querySelectorAll('tr[data-roi]'));
  assert(rows().length===3 && rows().every(row=>row.querySelector('input[type=checkbox]').checked) && /2 slides to compare/.test(tools().textContent),
    'One row per cell table with its slide, all ticked at first');
  const headers=()=>Array.from(tools().querySelectorAll('.studio-correction-matches th')).map(th=>th.textContent);
  assert(headers().includes('Slide ID'), 'Slide ID is shown for correction between slides');
  unit.value='(use batch column as-is)'; unit.dispatchEvent(new Event('change'));
  assert(!headers().includes('Slide ID'), 'The unit column follows Correct between');
  unit.value='slide'; unit.dispatchEvent(new Event('change'));
  const matches=tools().querySelector('.studio-correction-matches');
  assert(matches.scrollWidth<=matches.clientWidth+1, 'Image names, Slide IDs and state fit a narrow panel');
  assert(!tools().querySelector('[aria-label="Marker"]') && !/Colour the cells/.test(tools().textContent),'No colour section: Layers colours the cells');
  const s2=rows().find(row=>row.dataset.roi==='S2_roi_1');
  s2.querySelector('input[type=checkbox]').click();
  await until(()=>/1 slide to compare/.test(tools().textContent));
  assert(button(document.querySelector('.studio-runbar'),'Selected').disabled, 'Two ROIs on one slide cannot start a between-slide correction');
  const run=button(document.querySelector('.studio-runbar'),'Selected');
  assert(run && !button(document.querySelector('.studio-runbar'),'This image'),'Correct: Selected only (the correction compares tables, so there is no single-image run)');
  rows().find(row=>row.dataset.roi==='S2_roi_1').querySelector('input[type=checkbox]').click();
  await until(()=>/2 slides to compare/.test(tools().textContent));
  const norm=tools().querySelector('[aria-label="SOM normalization"]'); norm.value='rank'; norm.dispatchEvent(new Event('change',{bubbles:true}));
  await until(async()=> (await (await fetch('/spacemap-viewer/api/studio/browser-check/correction')).json()).settings.norm_method==='rank');
  assert(true, 'Normalization autosaves before running');
  run.click();
  await until(()=>run.disabled);await until(()=>!run.disabled);
  assert(/saved as batch_corrected_merged_/.test(tools().textContent) && /2 slides/.test(tools().textContent),'The correction runs as a job between the slides: '+document.querySelector('.studio-runbar').textContent);
  const plan=await (await fetch('/spacemap-viewer/api/studio/browser-check/correction')).json();
  assert(plan.unit==='slide' && plan.settings.correction_mode==='cycombine' && plan.settings.som_iterations===2000,'The choices are saved in Tab 6’s settings');
  assert(rows().every(row=>/✓ Corrected/.test(row.textContent)), 'Saved results turn green before adoption');
  const previousMethod=tools().querySelector('[aria-label="Method"]');
  button(tools(),'Use this table in Studio').click();
  await until(()=>tools().querySelector('[aria-label="Method"]') && tools().querySelector('[aria-label="Method"]')!==previousMethod);
  const combined=Array.from(tools().querySelectorAll('.studio-sec > details')).find(d=>d.querySelector('summary').textContent.startsWith('Combined table'));
  assert(combined && !combined.open, 'Saved result checks start folded'); combined.open=true;
  await until(()=>tools().querySelector('.studio-value') && /batch_corrected_merged_/.test(tools().textContent) && /Measured between/.test(tools().textContent) && /slides \(2\)/.test(tools().textContent));
  assert(true,'The new table is Studio’s combined table, measured between slides');
  await until(()=>tools().querySelectorAll('tr[data-roi] .chip.ok').length>=3 && !!tools().querySelector('[title^="batch_corrected_merged_"]'));
  assert(rows().every(row=>/Corrected/.test(row.textContent)),'Each table in the combined table shows Corrected');
  // Each marker's histograms per slide, before and after, beside the image.
  const figure=()=>document.querySelector('#studio-figures [data-figure="batches-histograms"]');
  await until(()=>figure() && figure().querySelectorAll('svg.studio-hist').length===2);
  const lines=figure().querySelectorAll('svg.studio-hist')[1].querySelectorAll('polyline');
  assert(lines.length===2 && figure().querySelectorAll('.studio-hist-unit').length===2,'One line per slide, as measured and corrected');
  figure().querySelector('.studio-hist-unit').click();
  assert(figure().querySelector('.studio-hist-unit').getAttribute('aria-pressed')==='true' && figure().querySelector('svg.studio-hist polyline[opacity="0.12"]'),'Clicking a slide picks it out');
  const row=Array.from(tools().querySelectorAll('.studio-measures .studio-measure')).find(b=>/^M1/.test(b.textContent));row.click();
  assert(figure().querySelector('[aria-label="Histogram marker"]').value==='M1' && /arcsinh\(M1/.test(figure().textContent),'Clicking a marker in How well it worked shows its histograms');
  const oldInput=tools().querySelector('[aria-label="Cell tables folder"]').value;
  const input=tools().querySelector('[aria-label="Cell tables folder"]'); input.value=oldInput+'/empty'; input.dispatchEvent(new Event('change'));
  await until(()=>tools().querySelector('[aria-label="Cell tables folder"]')?.value.endsWith('/empty') && tools().textContent.includes('No table'));
  assert(rows().length===3 && rows().every(row=>row.querySelector('input').disabled) && button(document.querySelector('.studio-runbar'),'Selected').disabled, 'Unmatched images stay visible and cannot run');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,900)));}
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
    (tables / 'empty').mkdir()
    rng = np.random.default_rng(5)
    for roi, bright in (("S1_roi_1", 1.0), ("S1_roi_2", 1.0), ("S2_roi_1", 1.8)):
        (images / (roi + '.ome.tiff')).touch()
        n = 300
        kind = rng.random(n) < 0.5
        pd.DataFrame(
            {
                "label": np.arange(1, n + 1),
                "area": rng.uniform(40, 60, n),
                "M0": bright * np.where(kind, 400, 40) * rng.lognormal(0, 0.2, n),
                "M1": bright * np.where(kind, 30, 300) * rng.lognormal(0, 0.2, n),
            }
        ).to_csv(tables / f"{roi}.ome_cell_features.csv", index=False)


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
