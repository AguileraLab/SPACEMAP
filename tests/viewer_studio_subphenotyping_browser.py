"""Browser-check Studio's Sub-phenotyping method on a disposable project in headless Firefox.

Settings in Tab 7's subphenotyping_params.json, Find markers as a job (its table ticked per phenotype and kept
when the panel is opened again), the split as a job, its sub-clusters in the cluster explorer, then named.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_subphenotyping_browser.py
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
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[aria-label="Phenotyping method"]'));
  const picker=document.querySelector('[aria-label="Phenotyping method"]');
  const open=method=>{picker.value=method;picker.dispatchEvent(new Event('change'));};
  assert(!picker.querySelector('option[value=subphenotyping]').textContent.includes('classic'),'Sub-phenotyping has Studio controls');
  open('subphenotyping');
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Split T cell"]') && tools().querySelector('[aria-label="Sub-phenotyping marker M3"]'));
  assert(tools().querySelector('[aria-label="Phenotype column to split"]').value==='phenotype_parent' && tools().querySelector('[aria-label="Split T cell"]').checked &&
    !tools().querySelector('[aria-label="Split Tumour"]').checked,'Tab 7’s saved column and phenotypes are chosen');
  assert(/Tumour \(100\)/.test(tools().textContent) && tools().textContent.includes('Saved'),'Each phenotype shows its cells; opening changes nothing');
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=String(value);input.dispatchEvent(new Event('change'));};
  set('Merge to',2);
  const gpu=tools().querySelector('[aria-label="Use the GPU"]');if(gpu.checked)gpu.click();
  const jobs=async()=>{const found=await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json();if(!found.jobs)throw Error('Jobs: '+found.error);return found.jobs;};
  button(document.querySelector('.studio-runbar'),'Find markers').click();
  await until(async()=>tools().querySelector('[aria-label="Use M3 for T cell"]') || ((await jobs()).find(job=>job.kind==='subphenotype')||{}).status==='failed');
  const found=(await jobs()).find(job=>job.kind==='subphenotype');
  assert(found.status==='done' && found.result.discovery['T cell'].cells===300,'Find markers runs as a job inside T cell only: '+found.message);
  // Keep the two markers that split the T cells (whatever was suggested).
  ['M0','M1','M2','M3'].forEach(name=>{const tick=tools().querySelector('[aria-label="Use '+name+' for T cell"]');if(tick.checked!==['M2','M3'].includes(name))tick.click();});
  assert(/T cell: 2 of 4 markers kept/.test(tools().textContent) && tools().textContent.includes('Not saved'),'The kept markers can be changed and show as not saved');
  button(document.querySelector('.studio-runbar'),'Run sub-phenotyping').click();
  await until(async()=>tools().querySelector('[aria-label="Phenotype of T_cell_1"]') || (await jobs()).filter(job=>job.kind==='subphenotype').some(job=>job.status==='failed'));
  const run=(await jobs()).find(job=>job.kind==='subphenotype' && job.result?.mode==='run');
  assert(run && run.status==='done' && run.result.column==='phenotype_sub_clusters' && run.result.clusters.length===2,'The split runs as a job: '+(run||{}).message);
  assert(JSON.stringify(run.result.markers_used['T cell'].slice().sort())==='["M2","M3"]' && run.result.cells===300,'The split uses the markers kept for T cell');
  const kept=await (await fetch('/spacemap-viewer/api/studio/browser-check/subpheno_settings')).json();
  assert(JSON.stringify(kept.values.chosen_markers['T cell'].slice().sort())==='["M2","M3"]' && kept.values.n_meta_clusters===2,'The settings are saved in Tab 7’s sub-phenotyping file');
  await until(()=>document.querySelector('#studio-figures .studio-heatmap .row'));
  const cells=state.layers.find(layer=>layer.type==='cells');
  tools().querySelector('[data-group="T_cell_0"]').click();
  await until(()=>cells.color_by==='phenotype_sub_clusters' && /Where: 2 ROIs/.test(tools().textContent));
  assert(true,'A sub-cluster shows on the image, with where it is');
  tools().querySelectorAll('[aria-label^="Phenotype of T_cell"]').forEach((input,index)=>{input.value=index?'T cell M3+':'T cell M2+';input.dispatchEvent(new Event('change'));});
  button(tools(),'Save names').click();
  for(let i=0;i<600;i++){const named=(await jobs()).find(job=>job.kind==='name_clusters');if(named && /done|failed/.test(named.status))break;await delay(100);}
  const named=(await jobs()).find(job=>job.kind==='name_clusters');
  assert(named.status==='done' && named.result.column==='phenotype_sub','The names are saved in a new column: '+named.message);
  // Opened again: the markers kept for T cell stay chosen and nothing shows as changed.
  open('leiden');await until(()=>tools().querySelector('[aria-label="Resolution"]'));
  open('subphenotyping');
  await until(()=>tools().querySelector('[aria-label="Sub-phenotyping marker M3"]') && /Run uses the markers kept earlier for T cell \(2\)/.test(tools().textContent));
  await delay(500);
  assert(!tools().textContent.includes('Not saved'),'Opening the panel again keeps the saved markers');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,900)));}
},100);
</script>
"""


def setup(root):
    """300 T cells in two states (M2 high or M3 high) and 100 tumour cells, 360 of them in a second ROI; Tab 7's settings."""
    import json

    import numpy as np
    import pandas as pd

    table = pd.read_csv(root / "cells.csv")
    extra = pd.concat([table] * 9, ignore_index=True)
    extra["batch_id"], extra["cell_id"], extra["label"] = "roi2", np.arange(40, 400), np.tile(np.arange(1, 41), 9)
    table = pd.concat([table, extra], ignore_index=True)
    rng = np.random.default_rng(3)
    state = np.arange(len(table)) % 2
    tumour = np.arange(len(table)) % 4 == 3
    table["phenotype_parent"] = np.where(tumour, "Tumour", "T cell")
    table["M2"] = np.where(state == 0, 400, 30) + rng.normal(0, 15, len(table))
    table["M3"] = np.where(state == 1, 400, 30) + rng.normal(0, 15, len(table))
    table.loc[tumour, ["M2", "M3"]] = 30 + rng.normal(0, 15, (int(tumour.sum()), 2))
    table.to_csv(root / "cells.csv", index=False)
    cache = root / "outputs/phenotyping/cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "subphenotyping_params.json").write_text(
        json.dumps({"parent_col": "phenotype_parent", "parent_values": ["T cell"], "markers": ["M0", "M1", "M2", "M3"]})
    )


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
