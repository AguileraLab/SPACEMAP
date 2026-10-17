"""Browser-check Studio's VORTEX method on a disposable project in headless Firefox.

Settings saved in Tab 7's vortex_params.json, VORTEX run as a job, its groups in the explorer, then named.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_vortex_browser.py
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
  assert(!picker.querySelector('option[value=vortex]').textContent.includes('classic'),'VORTEX has Studio controls');
  picker.value='vortex';picker.dispatchEvent(new Event('change'));
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Target clusters"]') && tools().querySelector('[aria-label="VORTEX marker M3"]'));
  assert(['M0','M2','M3'].every(name=>tools().querySelector('[aria-label="VORTEX marker '+name+'"]').checked),'Tab 7’s saved VORTEX markers are ticked');
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=String(value);input.dispatchEvent(new Event('change'));};
  const choose=tools().querySelector('[aria-label="Choose K"]');if(choose.checked)choose.click();
  const gpu=tools().querySelector('[aria-label="GPU"]');if(gpu.checked)gpu.click();
  set('K',8);set('Cells used',1000);set('Save as','phenotype_vortex_check');
  const dim=tools().querySelector('[aria-label="Dim cells apart"]');if(dim.checked)dim.click();
  // Tab 7's normalisation, in its order: Z-score each cell on, summarised in one line.
  tools().querySelector('[aria-label="Z-score each cell"]').click();
  assert(/Transformation: .*Z-score each marker.*→ Z-score each cell/.test(tools().textContent),'The normalisation chain is listed in Tab 7’s order');
  assert(tools().textContent.includes('Not saved'),'Changed settings show as not saved');
  button(document.querySelector('.studio-runbar'),'Run VORTEX').click();
  const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  await until(async()=>tools().querySelector('[aria-label^="Phenotype of Cluster"]') || ((await jobs()).find(job=>job.kind==='vortex')||{}).status==='failed');
  const run=(await jobs()).find(job=>job.kind==='vortex');
  const kept=await (await fetch('/spacemap-viewer/api/studio/browser-check/clustering/vortex')).json();
  assert(kept.values.norm_z2===true && kept.values.norm_z1===true,'The normalisation is saved in Tab 7’s VORTEX settings');
  assert(run.status==='done' && run.result.column==='phenotype_vortex_check_clusters' && run.result.clusters.length>=2,'VORTEX runs as a job and saves its groups: '+run.message);
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/clustering/vortex')).json();
  assert(saved.values.vortex_k===8 && saved.values.auto_select_k===false,'The settings are saved in Tab 7’s VORTEX file');
  await until(()=>document.querySelector('#studio-figures .studio-heatmap .row'));
  const cells=state.layers.find(layer=>layer.type==='cells');
  const second=run.result.clusters[1].name;
  tools().querySelector('[data-group="'+second+'"]').click();
  await until(()=>cells.color_by==='phenotype_vortex_check_clusters' && cells.hidden.size===run.result.clusters.length-1 && /Where: 1 ROI/.test(tools().textContent));
  assert(true,'A group shows on the image, with where it is');
  tools().querySelectorAll('[aria-label^="Phenotype of Cluster"]').forEach((input,index)=>{input.value=index?'Tumor':'T cell';input.dispatchEvent(new Event('change'));});
  button(tools(),'Save names').click();
  for(let i=0;i<600;i++){const found=(await jobs()).find(job=>job.kind==='name_clusters');if(found && /done|failed/.test(found.status))break;await delay(100);}
  const named=(await jobs()).find(job=>job.kind==='name_clusters');
  assert(named.status==='done' && named.result.column==='phenotype_vortex_check','The names are saved in a new column: '+named.message);
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(0,900)));}
},100);
</script>
"""


def setup(root):
    """Two populations (M2 high, then M3 high) in the harness's 40 cells, and Tab 7's saved VORTEX markers."""
    import json

    import pandas as pd

    table = pd.read_csv(root / "cells.csv")
    table["M2"] = [400 if i < 20 else 20 + i for i in range(len(table))]
    table["M3"] = [20 + i if i < 20 else 400 for i in range(len(table))]
    table.to_csv(root / "cells.csv", index=False)
    cache = root / "outputs/phenotyping/cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "vortex_params.json").write_text(json.dumps({"markers": ["M0", "M2", "M3"]}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
