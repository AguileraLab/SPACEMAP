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
  assert(['1. Choose cells','2. Find subsets','3. Name and save'].every(t=>tools().textContent.includes(t)), 'The workflow has three ordered sections');
  button(tools(),'Add filter').click();
  set('Cell filter 1 column','tumor');
  await until(()=>tools().querySelector('[aria-label="Cell filter 1 value"] option[value="1"]'));
  set('Cell filter 1 value','1');
  await until(()=>tools().querySelector('[data-subpheno-selection]').textContent==='240 cells selected');
  assert(true,'Region filtering previews the selected parent cells');
  await until(()=>document.querySelector('[aria-label="Process image roi2"]'));
  const firstImage=[...document.querySelectorAll('[aria-label^="Process image "]')].find(e=>e.getAttribute('aria-label')!=='Process image roi2');
  firstImage.click();
  await until(()=>tools().querySelector('[data-subpheno-selection]').textContent==='210 cells selected');
  assert(true,'The shared image checkboxes restrict the cell-count preview');
  const preprocessing=tools().querySelector('[data-subpheno-preprocessing]');
  assert(!preprocessing.open,'Preprocessing controls start collapsed');
  preprocessing.open=true;
  assert(tools().querySelector('[aria-label="Sub-phenotyping transformation"]').value==='auto','Raw markers default to arcsinh');
  set('Normalize within','slide');
  set('Sub-phenotyping arcsinh cofactor',10);
  const model=tools().querySelector('[aria-label="Sub-phenotyping model"]');
  assert(model.options.length>1,'Saved models are available in one dropdown');
  set('Sub-phenotyping model',model.options[1].value);
  set('Sub-phenotyping reference labels','phenotype_ref1');
  await until(()=>tools().querySelector('[aria-label="Suggest Tumour"]'));
  tools().querySelector('[aria-label="Limit model suggestions"]').click();
  await until(()=>tools().querySelector('[aria-label="Suggest Tumour"]')?.disabled===false);
  tools().querySelector('[aria-label="Suggest Tumour"]').click();
  set('Merge to',2);
  const gpu=tools().querySelector('[aria-label="Use the GPU"]');if(gpu.checked)gpu.click();
  const jobs=async()=>{const found=await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json();if(!found.jobs)throw Error('Jobs: '+found.error);return found.jobs;};
  const figures=tools().querySelector('[data-marker-figures]');figures.open=true;
  button(figures,'None').click();tools().querySelector('[aria-label="Generate Marker review"]').click();
  button(figures,'Generate selected').click();
  await until(()=>document.querySelector('[data-marker-review] [aria-label="Review use M3"]'));
  const review=document.querySelector('[data-marker-review]');
  assert(!review.querySelector('[aria-label="Review use M3"]').checked,'Review does not automatically choose clustering inputs');
  ['M2','M3'].forEach(m=>review.querySelector('[aria-label="Review use '+m+'"]').click());
  button(review,'Use selected for clustering').click();
  await until(()=>/T cell: 2 of 4 markers for clustering/.test(tools().textContent));
  assert(true,'Explicit marker choices from review are saved for the selected phenotype');
  button(document.querySelector('.studio-runbar'),'Run sub-phenotyping').click();
  await until(async()=>tools().querySelector('[aria-label="Phenotype of T_cell_1"]') || (await jobs()).filter(job=>job.kind==='subphenotype').some(job=>job.status==='failed'));
  const run=(await jobs()).find(job=>job.kind==='subphenotype' && job.result?.mode==='run');
  assert(run && run.status==='done' && run.result.column==='phenotype_sub_clusters' && run.result.clusters.length===2,'The split runs as a job: '+(run||{}).message);
  assert(JSON.stringify(run.result.markers_used['T cell'].slice().sort())==='["M2","M3"]' && run.result.cells===210,'The split uses the markers kept for the filtered T cells');
  assert(run.result.preprocessing.cofactor===10&&run.result.preprocessing.normalization_col==='slide'&&run.result.selection.images.join()==='roi2','Preprocessing and image selection reach the worker');
  assert(run.result.guidance.model_id===model.value && run.result.guidance.reference_column==='phenotype_ref1','The selected saved model and manual labels guide naming');
  assert(run.result.guidance.allowed.length===2&&!run.result.guidance.allowed.includes('Tumour'),'Only requested model types are offered');
  assert(tools().textContent.includes('Sample:') && tools().textContent.includes('Manual labels:'),'Model samples and manual counts are shown separately');
  assert([...tools().querySelectorAll('[aria-label^="Phenotype of T_cell"]')].every(i=>!i.value),'Suggestions do not assign names automatically');
  const kept=await (await fetch('/spacemap-viewer/api/studio/browser-check/subpheno_settings')).json();
  assert(JSON.stringify(kept.values.chosen_markers['T cell'].slice().sort())==='["M2","M3"]' && kept.values.n_meta_clusters===2,'The settings are saved in Tab 7’s sub-phenotyping file');
  await until(()=>document.querySelector('#studio-figures .studio-heatmap .row'));
  const heat=()=>document.querySelector('[data-figure="subphenotype"]');
  const colourRange=heat().querySelector('[aria-label="Heatmap colour range"]');
  assert(colourRange.value==='auto','Subset heatmap contrast starts fitted to its actual values');
  const beforeValues=[...heat().querySelectorAll('.row rect title')].map(e=>e.textContent);
  const beforeColours=[...heat().querySelectorAll('.row rect[fill]')].map(e=>e.getAttribute('fill')).join();
  colourRange.value='2.5';colourRange.dispatchEvent(new Event('change'));
  assert(beforeColours!==[...heat().querySelectorAll('.row rect[fill]')].map(e=>e.getAttribute('fill')).join(),'Colour range changes the heatmap contrast');
  assert(JSON.stringify(beforeValues)===JSON.stringify([...heat().querySelectorAll('.row rect title')].map(e=>e.textContent))&&heat().textContent.includes('+2.5'),'Values stay unchanged and the legend shows the chosen range');
  const realFetch=window.fetch;let figureRequest;
  window.fetch=async(url,options)=>{if(String(url).endsWith('/figure'))figureRequest=JSON.parse(options.body);return realFetch(url,options);};
  button(heat(),'Save figure').click();
  await until(()=>figureRequest&&button(heat(),'Save figure').disabled===false);
  window.fetch=realFetch;
  assert(figureRequest.options.vmin===-2.5&&figureRequest.options.vmax===2.5,'Export uses the displayed colour range');
  const cells=state.layers.find(layer=>layer.type==='cells');
  tools().querySelector('[data-group="T_cell_0"]').click();
  await until(()=>cells.color_by==='phenotype_sub_clusters' && /Where: 1 ROI/.test(tools().textContent));
  assert(true,'A sub-cluster shows on the image, with where it is');
  tools().querySelectorAll('[aria-label^="Phenotype of T_cell"]').forEach((input,index)=>{input.value=index?'T cell M3+':'T cell M2+';input.dispatchEvent(new Event('change'));});
  button(tools(),'Save names').click();
  for(let i=0;i<600;i++){const named=(await jobs()).find(job=>job.kind==='name_clusters');if(named && /done|failed/.test(named.status))break;await delay(100);}
  const named=(await jobs()).find(job=>job.kind==='name_clusters');
  assert(named.status==='done' && named.result.column==='phenotype_sub','The names are saved in a new column: '+named.message);
  // Opened again: the markers kept for T cell stay chosen and nothing shows as changed.
  open('leiden');await until(()=>tools().querySelector('[aria-label="Resolution"]'));
  open('subphenotyping');
  await until(()=>tools().querySelector('[aria-label="Sub-phenotyping marker M3"]') && /T cell: 2 of 4 markers for clustering/.test(tools().textContent));
  await delay(500);
  assert(!tools().textContent.includes('Not saved'),'Opening the panel again keeps the saved markers');
  assert(tools().querySelector('[aria-label="Normalize within"]').value==='slide'&&tools().querySelector('[aria-label="Sub-phenotyping arcsinh cofactor"]').value==='10','Preprocessing choices persist');
  const selected=await (await fetch('/spacemap-viewer/api/studio/browser-check/subpheno_settings')).json();
  assert(selected.values.filters[0].column==='tumor'&&selected.values.model_id===model.value,'Reopening keeps region and model choices');
  // The reported regression: switch corrected to raw, then Run without Find markers.
  set('Sub-phenotyping marker values','corrected');
  ['M4','M5','M6','M7'].forEach(m=>tools().querySelector('[aria-label="Sub-phenotyping marker '+m+'"]').click());
  set('Sub-phenotyping marker values','raw');
  assert(tools().querySelectorAll('[aria-label^="Sub-phenotyping marker M"]:checked').length===8&&tools().textContent.includes('Clustering: all 8 selected markers.'),'Switching to raw preserves all eight checked marker names and previews the actual inputs');
  set('Sub-phenotype result column','phenotype_eight');
  // Hold the follow-up names and table refreshes indefinitely. Jobs already
  // contains the new profile, so neither request should delay its display.
  let releaseRefresh, doneAt=0, heldNames=0, released=false;
  const refreshGate=new Promise(resolve=>{releaseRefresh=()=>{released=true;resolve();};});
  window.fetch=async(url,options)=>{
    const response=await realFetch(url,options), path=String(url);
    if(path.endsWith('/jobs')){
      const value=await response.clone().json();
      if(value.jobs?.some(j=>j.status==='done'&&j.result?.column==='phenotype_eight_clusters'))doneAt ||= performance.now();
    }
    if(doneAt && (path.includes('/cluster_names?')||path.endsWith('/phenotyping_inputs'))){
      if(path.includes('/cluster_names?'))heldNames++;
      await refreshGate;
    }
    return response;
  };
  button(document.querySelector('.studio-runbar'),'Run sub-phenotyping').click();
  await until(async()=>(await jobs()).some(j=>j.kind==='subphenotype'&&j.result?.column==='phenotype_eight_clusters'));
  const eight=(await jobs()).find(j=>j.result?.column==='phenotype_eight_clusters');
  assert(eight.status==='done'&&eight.result.markers_used['T cell'].length===8&&eight.result.markers_used['T cell'].every(m=>!m.endsWith('_corr')),'Run uses all eight raw markers without automatic filtering');
  assert(eight.result.profile.columns.length===8,'The saved heatmap contains all eight markers');
  while(heat()?.querySelectorAll('.studio-heatmap .column').length!==8 && performance.now()-doneAt<5000)await delay(25);
  assert(heat()?.querySelectorAll('.studio-heatmap .column').length===8,'The rerun heatmap updates within one Jobs polling interval, without waiting for table or name reloads');
  await until(()=>heldNames>0);
  assert(!released,'The new heatmap and naming list are usable while metadata requests remain pending');
  const draft=tools().querySelector('[aria-label="Phenotype of T_cell_0"]');
  draft.value='Current draft';draft.dispatchEvent(new Event('input'));draft.dispatchEvent(new Event('change'));
  releaseRefresh();window.fetch=realFetch;
  await delay(500);
  assert(draft.value==='Current draft'&&heat().textContent.includes('Current draft'),'Delayed metadata keeps edits made to the new run');
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
    table['tumor'] = (np.arange(len(table)) < 320).astype(int)
    table['slide'] = np.where(np.arange(len(table)) < 200, 'Slide1', 'Slide2')
    table['phenotype_ref1'] = np.where(tumour, 'Tumour', np.where(state == 0, 'T M2+', 'T M3+'))
    table["M2"] = np.where(state == 0, 400, 30) + rng.normal(0, 15, len(table))
    table["M3"] = np.where(state == 1, 400, 30) + rng.normal(0, 15, len(table))
    table.loc[tumour, ["M2", "M3"]] = 30 + rng.normal(0, 15, (int(tumour.sum()), 2))
    for name in ['M4', 'M5', 'M6', 'M7']:
        table[name] = 5.
    for name in [f'M{i}' for i in range(8)]:
        table[name + '_corr'] = np.arcsinh(table[name] / 5.)
    table.to_csv(root / "cells.csv", index=False)
    from spacemap import dataio
    from spacemap_gui.studio import reference_models, tasks
    dataio.ensure_fast_copy(root / 'cells.csv', force=True)
    settings = {'project_root': str(root), 'table': str(root / 'cells.csv'), 'training_column': 'phenotype_ref1',
                'roi_column': 'batch_id', 'label_column': 'label', 'markers': ['M2', 'M3']}
    settings['revision'] = tasks.source_revision(settings)
    reference_models.train(settings, root / 'trained', lambda *args: None)
    cache = root / "outputs/phenotyping/cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "subphenotyping_params.json").write_text(
        json.dumps({"parent_col": "phenotype_parent", "parent_values": ["T cell"], "markers": ["M0", "M1", "M2", "M3"]})
    )


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
