"""Browser-check Studio's Spatial analysis: cell-cell interactions, and classic Tab 8's saved triads and screening.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_interactions_browser.py
"""

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
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[data-step=tab8]'));
  document.querySelector('[data-step=tab8]').click();
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('.studio-stage-tabs'));
  Array.from(tools().querySelectorAll('.studio-stage-tabs button')).find(b=>b.textContent==='Cell–cell interactions').click();
  await until(()=>tools().querySelector('[aria-label="Interaction phenotype column"]') && /No test yet/.test(tools().textContent));
  const phenotypes=tools().querySelector('[aria-label="Interaction phenotype column"]');phenotypes.value='phenotype_ml_consensus';phenotypes.dispatchEvent(new Event('change'));
  const shuffles=tools().querySelector('[aria-label="Shuffles"]');shuffles.value='60';shuffles.dispatchEvent(new Event('change'));
  const small=tools().querySelector('[aria-label="Smallest region for interactions"]');small.value='5';small.dispatchEvent(new Event('change'));
  await until(()=>tools().querySelector('tr[data-roi="roi"]')&&/Test:/.test(document.querySelector('.studio-runbar').textContent));
  button(document.querySelector('.studio-runbar'),'Selected').click();
  await until(()=>/Tick the images first/.test(document.querySelector('.studio-runbar').textContent));
  tools().querySelector('[aria-label="Select all"]').click();
  button(document.querySelector('.studio-runbar'),'Selected').click();
  await until(()=>document.querySelector('#studio-figures [data-figure="interactions"] .studio-heatmap .row'));
  const settings=await (await fetch('/spacemap-viewer/api/studio/browser-check/interaction_settings')).json();
  assert(settings.values.n_permutations===60 && settings.values.phenotype_col==='phenotype_ml_consensus','The settings are saved in Tab 8’s keys');
  const job=(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs.find(item=>item.kind==='cell_interactions');
  const same=job.result.pairs.find(pair=>pair.a==='A' && pair.b==='A'), mixed=job.result.pairs.find(pair=>pair.a==='A' && pair.b==='B');
  assert(same.z>2 && mixed.z<-2,'Cells of one half are neighbours more than chance, of the two halves less: '+JSON.stringify(job.result.pairs));
  const pairButton=Array.from(tools().querySelectorAll('.studio-measure')).find(item=>/^A – B/.test(item.textContent));pairButton.click();
  const cells=state.layers.find(layer=>layer.type==='cells');
  await until(()=>cells.color_by==='phenotype_ml_consensus' && cells.hidden.size===0 && /Showing A and B/.test(tools().textContent));
  const aa=Array.from(tools().querySelectorAll('.studio-measure')).find(item=>/^A – A/.test(item.textContent));aa.click();
  await until(()=>cells.hidden.has('B') && !cells.hidden.has('A'));
  assert(true,'A pair shows its phenotypes on the image');
  // Classic Tab 8's saved results: a triad type drawn as a density map; screening's findings and figures.
  Array.from(tools().querySelectorAll('.studio-stage-tabs button')).find(b=>b.textContent==='Triads').click();
  await until(()=>/triad_instances_local\.csv · 5 triads/.test(tools().textContent));
  const triad=Array.from(tools().querySelectorAll('.studio-measure')).find(item=>/^A · A · B/.test(item.textContent));
  assert(triad && /3$/.test(triad.textContent.trim()),'Each triad type is listed with its count');
  triad.click();
  await until(()=>state.layers.some(layer=>layer.type==='density' && (layer.fvals||[]).includes('A::A::B')));
  assert(true,'A triad type is drawn on the image as a density map');
  Array.from(tools().querySelectorAll('.studio-stage-tabs button')).find(b=>b.textContent==='Feature screening').click();
  await until(()=>/3 tests/.test(tools().textContent));
  assert(/3 tests/.test(tools().textContent) && /1 with q < 0.05/.test(tools().textContent) && tools().querySelectorAll('.studio-counts tr').length>=4,'The screening run’s findings are listed');
  tools().querySelector('.studio-thumb').click();
  await until(()=>document.querySelector('#studio-figures [data-figure="screening-0"] img'));
  assert(true,'A screening figure opens beside the image');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(-700)));}
},100);
</script>
"""


def setup(root):
    """A triad table for the ROI and a feature screening run, as classic Tab 8 saves them."""
    import pandas as pd
    from PIL import Image

    triads = root / "outputs/spatial_analysis/triad"
    triads.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "batch_id": "roi",
            "triad_label": ["A::A::B"] * 3 + ["A::B::B"] * 2,
            "triad_centroid_x": [60.0, 100, 140, 180, 220],
            "triad_centroid_y": [60.0, 100, 140, 180, 220],
        }
    ).to_csv(triads / "triad_instances_local.csv", index=False)
    run = root / "outputs/spatial_feature_screening/feature_screening_20260928_120000"
    (run / "tables").mkdir(parents=True, exist_ok=True)
    (run / "figures").mkdir(exist_ok=True)
    pd.DataFrame(
        {"feature_label": ["A density", "B density", "A–B contacts"], "endpoint_label": "OS", "p_value": [0.3, 0.001, 0.04], "fdr_q": [0.3, 0.01, 0.06]}
    ).to_csv(run / "tables" / "screening_results_all.csv", index=False)
    Image.new("RGB", (600, 300), "white").save(run / "figures" / "forest_plot.png")


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
