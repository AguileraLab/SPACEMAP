"""Browser-check Studio's SOM method on a disposable project in headless Firefox.

Settings saved in Tab 7's som_params.json, SOM run as a job, its groups in the explorer, then named.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_som_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<3600;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
window.addEventListener('unhandledrejection',e=>fetch('/result?error='+encodeURIComponent(String(e.reason))));
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[aria-label="Phenotyping method"]'));
  const picker=document.querySelector('[aria-label="Phenotyping method"]');
  assert(!picker.querySelector('option[value=som]').textContent.includes('classic'),'SOM has Studio controls');
  picker.value='som';picker.dispatchEvent(new Event('change'));
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Meta-clusters"]') && tools().querySelector('[aria-label="SOM marker M3"]'));
  assert(['M0','M2','M3'].every(name=>tools().querySelector('[aria-label="SOM marker '+name+'"]').checked),'Tab 7’s saved SOM markers are ticked');
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=String(value);input.dispatchEvent(new Event('change'));};
  set('Grid',2);set('Meta-clusters',2);set('Quality','fast');set('Cells used',1000);set('Save as','phenotype_som_check');
  const dim=tools().querySelector('[aria-label="Dim cells apart"]');if(dim.checked)dim.click();
  // Tab 7's normalisation, in its order: Z-score each cell on, summarised in one line.
  tools().querySelector('[aria-label="Z-score each cell"]').click();
  assert(/Transformation: .*Z-score each marker.*→ Z-score each cell/.test(tools().textContent),'The normalisation chain is listed in Tab 7’s order');
  assert(tools().textContent.includes('Not saved'),'Changed settings show as not saved');
  // Save once, then edit and switch methods: the draft must remain the live settings object.
  button(document.querySelector('.studio-runbar'),'Save settings').click();
  await until(()=>document.querySelector('.studio-runbar').textContent.includes('Saved into Tab 7'));
  tools().querySelector('[aria-label="Z-score each cell"]').click();
  picker.value='vortex';picker.dispatchEvent(new Event('change'));
  await until(()=>tools().querySelector('[aria-label="VORTEX K"]') || tools().querySelector('[aria-label="Target clusters"]'));
  assert(!!tools().querySelector('[aria-label="Overwrite existing results"]'), 'VORTEX has overwrite too');
  picker.value='som';picker.dispatchEvent(new Event('change'));
  await until(()=>tools().querySelector('[aria-label="SOM marker M3"]'));
  assert(!tools().querySelector('[aria-label="Z-score each cell"]').checked, 'Edits after saving survive switching methods');
  // Browser form restoration can set controls without a change event. Run must read what is shown.
  const flags={'Corrected values back to the measured scale':false,'Z-score each marker':true,
    'Z-score each cell':true,'Normal CDF (0 to 1)':true,'−log(1 − CDF)':true};
  Object.entries(flags).forEach(([label,value])=>{tools().querySelector('[aria-label="'+label+'"]').checked=value;});
  button(document.querySelector('.studio-runbar'),'Run SOM').click();
  const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  await until(async()=>tools().querySelector('[aria-label^="Phenotype of Cluster"]') || ((await jobs()).find(job=>job.kind==='som')||{}).status==='failed');
  const run=(await jobs()).find(job=>job.kind==='som');
  const kept=await (await fetch('/spacemap-viewer/api/studio/browser-check/clustering/som')).json();
  assert(!kept.values.norm_sinh && kept.values.norm_z2 && kept.values.norm_z1 && kept.values.norm_cdf && kept.values.norm_nlog,
    'The exact visible transformations are saved, even without change events');
  assert(run.result.profile.transformation.input==='Corrected (arcsinh)' && run.result.profile.transformation.steps.join('|')==='Z1|Z2|CDF|−log(1−CDF)',
    'The SOM heatmap reports the exact selected transformation and no inverse sinh');
  assert(run.status==='done' && run.result.column==='phenotype_som_check_clusters' && run.result.clusters.length===2,'SOM runs as a job and saves its groups: '+run.message);
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/clustering/som')).json();
  assert(saved.values.grid_dim===2 && saved.values.n_meta_target===2 && saved.values.mode==='fast','The settings are saved in Tab 7’s SOM file');
  await until(()=>document.querySelector('#studio-figures .studio-heatmap .row'));
  const cells=state.layers.find(layer=>layer.type==='cells');
  const second=run.result.clusters[1].name;  // meta-clusters keep the merge's numbers (0 and 3, say)
  tools().querySelector('[data-group="'+second+'"]').click();
  await until(()=>cells.color_by==='phenotype_som_check_clusters' && cells.hidden.size===1 && /Where: 1 ROI/.test(tools().textContent));
  assert(true,'A group shows on the image, with where it is');
  tools().querySelectorAll('[aria-label^="Phenotype of Cluster"]').forEach((input,index)=>{input.value=index?'Tumor':'T cell';input.dispatchEvent(new Event('change'));});
  button(tools(),'Save names').click();
  for(let i=0;i<600;i++){const found=(await jobs()).find(job=>job.kind==='name_clusters');if(found && /done|failed/.test(found.status))break;await delay(100);}
  const named=(await jobs()).find(job=>job.kind==='name_clusters');
  assert(named.status==='done' && named.result.column==='phenotype_som_check','The names are saved in a new column: '+named.message);
  await until(()=>!button(tools(),'Save names').disabled);
  tools().querySelectorAll('[aria-label^="Phenotype of Cluster"]').forEach(input=>{input.value='Revised';input.dispatchEvent(new Event('change'));});
  button(tools(),'Save names').click();
  await until(async()=>{const n=(await jobs()).find(j=>j.kind==='name_clusters');return n && n.id!==named.id && /done|failed/.test(n.status);});
  const edited=(await jobs()).find(j=>j.kind==='name_clusters');
  assert(edited.status==='done' && edited.result.column==='phenotype_som_check' && edited.result.phenotypes.Revised===40,
    'Saving names again updates the requested column, without a _2 suffix');
  const overwrite=tools().querySelector('[aria-label="Overwrite existing results"]');
  assert(overwrite && !overwrite.checked,'SOM overwrite is off by default');
  button(document.querySelector('.studio-runbar'),'Run SOM').click();
  await until(()=>document.querySelector('.studio-runbar').textContent.includes('tick Overwrite existing results'));
  assert((await jobs()).filter(j=>j.kind==='som').length===1,'An unchecked rerun preserves existing SOM results');
  overwrite.click();
  button(document.querySelector('.studio-runbar'),'Run SOM').click();
  await until(async()=>{const j=(await jobs()).find(j=>j.kind==='som');return j && j.id!==run.id && /done|failed/.test(j.status);});
  const replaced=(await jobs()).find(j=>j.kind==='som');
  assert(replaced.status==='done' && replaced.result.overwrite && replaced.result.columns.includes('phenotype_som_check'),
    'Explicit SOM overwrite replaces cluster membership and resets old names');
  assert(!overwrite.checked,'Overwrite resets after starting the run');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,900)));}
},100);
</script>
"""


def setup(root):
    """Two populations (M2 high, then M3 high) in the harness's 40 cells, and Tab 7's saved SOM markers."""
    import json

    import pandas as pd
    import numpy as np

    table = pd.read_csv(root / "cells.csv")
    table["M2"] = [400 if i < 20 else 20 + i for i in range(len(table))]
    table["M3"] = [20 + i if i < 20 else 400 for i in range(len(table))]
    for marker in ("M0", "M2", "M3"):
        table[marker + "_corr"] = np.arcsinh(table[marker] / 5.)
    table.to_csv(root / "cells.csv", index=False)
    cache = root / "outputs/phenotyping/cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "som_params.json").write_text(json.dumps({"markers": ["M0_corr", "M2_corr", "M3_corr"], "batch_id_col": ""}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
