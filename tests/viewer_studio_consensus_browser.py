"""Browser-check Studio's Consensus ML method on a disposable project in headless Firefox.

Settings in Tab 7's consensus_ml_params.json, the run as a job, the result's numbers, phenotypes and heatmap,
and the result columns waiting in Jobs > Saved results.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_consensus_browser.py
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
  assert(!picker.querySelector('option[value=consensus]').textContent.includes('classic'),'Consensus ML has Studio controls');
  picker.value='consensus';picker.dispatchEvent(new Event('change'));
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Combine phenotype_a"]') && tools().querySelector('[aria-label="Consensus marker M3"]'));
  assert(['phenotype_a','phenotype_b','phenotype_c'].every(name=>tools().querySelector('[aria-label="Combine '+name+'"]').checked),'Tab 7’s saved methods are ticked');
  assert(!tools().querySelector('[aria-label="Combine phenotype_ref1"]') && Array.from(tools().querySelector('[aria-label="Check against"]').options).some(o=>o.value==='phenotype_ref1'),
    'Reference labels never vote; they can check the result');
  const trees=tools().querySelector('[aria-label="Trees"]');trees.value='20';trees.dispatchEvent(new Event('change'));
  const cores=tools().querySelector('[aria-label="CPU cores"]');
  const backend=tools().querySelector('[aria-label="Consensus execution"]');
  assert(cores && cores.value==='0' && backend.value==='cpu','Compute defaults to CPU and Auto cores');
  assert(cores.closest('details') && !cores.closest('details').open,'Compute controls are compact and expandable');
  cores.closest('details').open=true;
  backend.value='auto';backend.dispatchEvent(new Event('change'));
  const capability=await (await fetch('/spacemap-viewer/api/studio/browser-check/consensus_settings')).json();
  if(!capability.hardware.gpu.available){
    assert(cores.closest('details').querySelector('summary').textContent.includes('CPU · '+capability.hardware.cpu.auto_workers+' cores · GPU unavailable'),
      'Unavailable GPU is shown as CPU with the actual Auto core budget before running');
  }
  assert(!tools().querySelector('[aria-label="GPU split bins"]').closest('.field').hidden,'GPU choice reveals split bins');
  backend.value='cpu';backend.dispatchEvent(new Event('change'));
  // Deliberately do not blur/change: Run must still read the current core count.
  cores.value='4';
  assert(tools().textContent.includes('Not saved'),'Changed settings show as not saved');
  const overwrite=tools().querySelector('[aria-label="Overwrite existing results"]');
  assert(overwrite && !overwrite.checked,'Consensus overwrite defaults off');
  button(document.querySelector('.studio-runbar'),'Run Consensus ML').click();
  await until(()=>document.querySelector('.studio-runbar').textContent.includes('tick Overwrite existing results'));
  assert(true,'A taken Consensus output is kept unless overwrite is checked');
  overwrite.click();
  button(document.querySelector('.studio-runbar'),'Run Consensus ML').click();
  const jobs=async()=>{const found=await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json();if(!found.jobs)throw Error('Jobs: '+found.error);return found.jobs;};
  await until(async()=>/Agreed on \d+%/.test(tools().textContent) || ((await jobs()).find(job=>job.kind==='consensus_ml')||{}).status==='failed');
  const run=(await jobs()).find(job=>job.kind==='consensus_ml');
  assert(run.status==='done' && run.result.column==='phenotype_ml_consensus','Consensus ML overwrites the selected output without adding a suffix: '+run.message);
  const kept=await (await fetch('/spacemap-viewer/api/studio/browser-check/consensus_settings')).json();
  assert(kept.values.n_estimators===20,'The settings are saved in Tab 7’s Consensus ML file');
  assert(kept.values.cpu_workers===4 && kept.values.execution_backend==='cpu','Core count reaches saved settings without blur');
  assert(run.result.execution.workers===Math.min(4,kept.hardware.cpu.available),'The job uses the saved core budget');
  await until(()=>document.querySelector('#studio-figures .studio-heatmap .row'));
  const heat=()=>document.querySelector('#studio-figures .studio-heatmap');
  const first=run.result.profile.rows[0];
  heat().querySelector('.row[data-row="'+first+'"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
  const cells=state.layers.find(layer=>layer.type==='cells');
  await until(()=>cells.color_by==='phenotype_ml_consensus' && cells.hidden.size===run.result.profile.rows.length-1);
  assert(true,'Clicking a phenotype shows only its cells on the image');
  const summary=()=>document.querySelector('[aria-label="Heatmap summary"]');
  assert(summary()?.value==='median','Consensus profiles default to Median');
  summary().value='mean';summary().dispatchEvent(new Event('change'));
  await until(()=>summary()?.value==='mean' && /Mean · Input marker values/.test(document.querySelector('#studio-figures').textContent));
  assert(heat().querySelector('.row.picked')?.dataset.row===first && cells.hidden.size===run.result.profile.rows.length-1,'Changing to Mean keeps the phenotype selected in the heatmap and image');
  assert((await jobs()).find(j=>j.id===run.id).result.profile.aggregation==='mean','Consensus Mean preference reaches the shared job result');
  document.querySelector('[data-tab=jobs]').click();
  const pane=document.querySelector('#studio-jobs');
  await until(()=>/phenotype_ml_consensus_refined/.test(pane.textContent));
  assert(/phenotype_ml_consensus_score/.test(pane.textContent) && button(pane,'Write into the CSV'),'The four result columns wait in Saved results');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,900)));}
},100);
</script>
"""


def setup(root):
    """Two populations (M2 high, then M3 high); three methods that mostly agree; Tab 7's saved Consensus ML settings."""
    import json

    import pandas as pd

    table = pd.read_csv(root / "cells.csv")
    table["M2"] = [400 if i < 20 else 20 + i for i in range(len(table))]
    table["M3"] = [20 + i if i < 20 else 400 for i in range(len(table))]
    truth = ["A" if i < 20 else "B" for i in range(len(table))]
    for method, wrong in (("phenotype_a", {3}), ("phenotype_b", {25}), ("phenotype_c", {7, 30})):
        table[method] = [("B" if call == "A" else "A") if i in wrong else call for i, call in enumerate(truth)]
    table.to_csv(root / "cells.csv", index=False)
    cache = root / "outputs/phenotyping/cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "consensus_ml_params.json").write_text(
        json.dumps(
            {
                "markers": ["M0", "M2", "M3"],
                "phenotyping_methods": ["phenotype_a", "phenotype_b", "phenotype_c"],
                "min_consensus_methods": 3,
                "min_training_cells": 5,
                "enable_filter": False,
                "exclude_phenotypes": [],
            }
        )
    )


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
