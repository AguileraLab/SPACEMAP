"""Browser-check Studio's Leiden method on a disposable project in headless Firefox.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_leiden_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<3600;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[aria-label="Phenotyping method"]'));
  const picker=document.querySelector('[aria-label="Phenotyping method"]');
  assert(!picker.querySelector('option[value=leiden]').textContent.includes('classic'),'Leiden has Studio controls');
  picker.value='leiden';picker.dispatchEvent(new Event('change'));
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Resolution"]') && tools().querySelector('[aria-label="Leiden marker M0"]'));
  await until(()=>tools().querySelector('[aria-label="Phenotyping combined table"]'));
  assert(tools().querySelector('.studio-step-body > .studio-sec .studio-sec-h').textContent.includes('Folders and tables'), 'Phenotyping folders and the selected table are at the top');
  assert(tools().querySelectorAll('tr[data-roi]').length===1 && tools().querySelector('[aria-label="Phenotyping Cell ID column"]').value==='cell_id', 'The combined table supplies ROI rows and cell identity');
  assert(tools().querySelector('[aria-label="Phenotyping ROI column"]').value==='batch_id' && tools().querySelector('[aria-label="Phenotyping Slide column"]').value==='slide', 'ROI and physical slide columns have explicit roles');
  await until(()=>tools().querySelector('[aria-label="Normalise within"]')?.options.length===4);
  const normalizationOptions=Array.from(tools().querySelector('[aria-label="Normalise within"]').options).map(o=>o.textContent);
  assert(normalizationOptions.includes('Each slide (slide)') && normalizationOptions.includes('Each value of slide_id') && new Set(normalizationOptions).size===normalizationOptions.length, 'Slide and slide_id never share a misleading label');
  // The markers are listed from the combined table, even before an image shows them; a saved set ticks them.
  const sets=tools().querySelector('[aria-label="Leiden marker set"]');
  assert(sets.options.length===2 && /phenotyping_markers \(2, Image viewer\)/.test(sets.options[1].textContent),'The viewer’s marker sets are offered');
  sets.value='0';sets.dispatchEvent(new Event('change'));
  assert(['M0','M1'].every(name=>tools().querySelector('[aria-label="Leiden marker '+name+'"]').checked) && /2 chosen/.test(tools().textContent),'Picking a set ticks its markers');
  assert(tools().querySelector('[aria-label="Leiden marker values"] option[value=corrected]').disabled,'Corrected values are offered only when the table has them');
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=String(value);input.dispatchEvent(new Event(input.type==='range'?'input':'change'));};
  set('Resolution',1);set('Cells used',1000);set('Leiden result column','phenotype_leiden_check');
  ['Use the GPU','Keep dim cells apart'].forEach(label=>{const tick=tools().querySelector('[aria-label="'+label+'"]');if(tick.checked)tick.click();});
  assert(tools().textContent.includes('Not saved'),'Changed settings show as not saved');
  button(document.querySelector('.studio-runbar'),'Run Leiden').click();
  await until(()=>tools().querySelector('[aria-label="Phenotype of Cluster 0"]') || /failed/.test(document.querySelector('#studio-jobs').textContent));
  const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  const run=(await jobs()).find(job=>job.kind==='leiden');
  assert(run.status==='done' && run.result.column==='phenotype_leiden_check_clusters','Leiden runs as a job and saves the clusters: '+run.message);
  await until(()=>Array.from(tools().querySelector('[aria-label="ROI result status"]').options).some(o=>o.value===run.result.column));
  const resultStatus=tools().querySelector('[aria-label="ROI result status"]');resultStatus.value=run.result.column;resultStatus.dispatchEvent(new Event('change'));
  assert(tools().querySelector('tr[data-roi] .chip.ok')?.textContent.includes('Processed'), 'Finished Leiden results appear as green processed indicators in the ROI list');
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/leiden_settings')).json();
  assert(saved.values.markers.length===2 && saved.values.resolution===1 && saved.values.use_gpu===false,'Run saved the settings into Tab 7\'s Leiden file first');
  const summary=()=>document.querySelector('[aria-label="Heatmap summary"]');
  await until(()=>summary());
  assert(summary().value==='median' && /Median · Measured values → Z1/.test(document.querySelector('#studio-figures').textContent),'Median is the default and the saved transformation is shown');
  summary().value='mean';summary().dispatchEvent(new Event('change'));
  await until(()=>summary()?.value==='mean' && /Mean · Measured values/.test(document.querySelector('#studio-figures').textContent));
  const meanRun=(await jobs()).find(j=>j.id===run.id);
  assert(meanRun.result.profile.aggregation==='mean' && JSON.stringify(meanRun.result.profile.raw)===JSON.stringify(run.result.profile.summaries.mean.raw),'Switching to Mean reads the saved transformed means');
  assert(JSON.stringify(meanRun.result.clusters.map(c=>[c.id,c.cells]))===JSON.stringify(run.result.clusters.map(c=>[c.id,c.cells])),'Summary changes preserve cluster membership');
  assert(document.querySelector('#studio-figures .studio-heatmap rect title')?.textContent.includes('Mean after the selected transformations'),'Hover labels distinguish transformed means from the colour z-score');
  summary().value='median';summary().dispatchEvent(new Event('change'));
  await until(()=>summary()?.value==='median' && /Median · Measured values/.test(document.querySelector('#studio-figures').textContent));
  const rows=tools().querySelectorAll('[aria-label^="Phenotype of Cluster"]');
  assert(rows.length===run.result.clusters.length && rows.length>1,'Every cluster can be named');
  await until(()=>document.querySelector('#studio-figures .studio-heatmap .row'));
  // The heatmap is redrawn in place (names, suggestions, pane width): always look it up again.
  const heat=()=>document.querySelector('#studio-figures .studio-heatmap');
  const row=name=>heat().querySelector('.row[data-row="'+name+'"]');
  assert(heat().querySelectorAll('.row').length===run.result.clusters.length && heat().querySelectorAll('.column').length===2,'The clusters heatmap is beside the image: a row per cluster, a column per marker');
  const axisX=()=>document.querySelector('[aria-label="Cluster heatmap X axis"]'),axisY=()=>document.querySelector('[aria-label="Cluster heatmap Y axis"]');
  const rowIds=()=>Array.from(heat().querySelectorAll('.row')).map(r=>r.dataset.row);
  const columnIds=()=>Array.from(heat().querySelectorAll('.column')).map(c=>c.textContent);
  const originalRows=rowIds(),originalColumns=columnIds();
  axisX().click();
  assert(JSON.stringify(rowIds())===JSON.stringify(originalRows)&&JSON.stringify(columnIds())===JSON.stringify(run.result.profile.column_order.map(i=>originalColumns[i])),'X clustering reorders markers without changing cluster rows');
  axisY().click();
  assert(JSON.stringify(rowIds())===JSON.stringify(run.result.profile.row_order.map(i=>originalRows[i])),'Y clustering uses the full saved profiles');
  axisX().click();
  assert(JSON.stringify(columnIds())===JSON.stringify(originalColumns)&&axisY().checked,'Y-only clustering leaves marker order unchanged');
  axisY().click();
  const bar=heat().querySelector('.heatmap-scale').getBBox(),lastSquare=heat().querySelector('.row rect:has(title)').getBBox();
  assert(bar.x>lastSquare.x+lastSquare.width&&heat().querySelector('.scale-label').textContent==='Z-score','The heatmap has a labelled colour scale on the right');
  assert(tools().querySelector('[aria-label="Figure data source"]').value==='run','Publication heatmaps default to the full current clustering run');
  tools().querySelector('[aria-label="Cluster figure Y axis"]').click();
  const realFigureFetch=window.fetch;let publicationRequest=null;
  window.fetch=async(url,options)=>{if(String(url).endsWith('/figure')&&options?.body)publicationRequest=JSON.parse(options.body);return realFigureFetch(url,options);};
  button(tools(),'Make figure').click();
  await until(()=>/Saved leiden_phenotype_heatmap\.png/.test(tools().textContent));
  assert(publicationRequest.source==='run'&&publicationRequest.job_id===run.id&&!publicationRequest.image_token&&!publicationRequest.selected_cells&&publicationRequest.options.cluster_x&&!publicationRequest.options.cluster_y,'Publication figure uses the full run, with independent X/Y controls and no image or selected-cell restriction');
  window.fetch=realFigureFetch;
  // Suggestions from the reference column (A and B in the test table) reach the list and the heatmap.
  await until(()=>/\d+ reference cells, 2 labels/.test(tools().textContent) && heat().querySelector('.note'));
  assert(tools().querySelectorAll('.studio-hints .studio-suggestion').length>0 && /Reference cells: /.test(tools().textContent),'Each cluster gets name suggestions and its reference cells');
  const narrow=heat().getBoundingClientRect().width;
  button(document.querySelector('#studio-figures'),'Large view').click();
  await until(()=>document.querySelector('#stage .studio-figure-large .studio-heatmap') && document.querySelector('#stage .studio-figure-large .studio-heatmap').getBoundingClientRect().width>narrow+40);
  assert(true,'Large view shows the heatmap over the image, wider');
  summary().value='mean';summary().dispatchEvent(new Event('change'));
  await until(()=>document.querySelector('#stage .studio-figure-large [aria-label="Heatmap summary"]')?.value==='mean');
  assert(document.querySelectorAll('svg[aria-label="Leiden clusters"]').length===1&&document.querySelector('#stage .studio-figure-large').textContent.includes('Mean ·'),'Changing summary in Large view replaces the old heatmap and keeps it enlarged');
  summary().value='median';summary().dispatchEvent(new Event('change'));
  await until(()=>document.querySelector('#stage .studio-figure-large [aria-label="Heatmap summary"]')?.value==='median');
  button(document.querySelector('#stage .studio-figure-large'),'Back to the pane').click();
  await until(()=>!document.querySelector('.studio-figure-large') && heat());
  button(tools(),'View data').click();
  await until(()=>dataView.source?.combined && dataView.result?.columns.includes(run.result.column));
  assert(dataView.result.total===40, 'Data reads the combined input and its new saved cluster column');
  document.querySelector('[aria-label="Only selected cells"]').click();
  row('Cluster 1').dispatchEvent(new MouseEvent('click',{bubbles:true}));
  const cells=state.layers.find(layer=>layer.type==='cells');
  await until(()=>cells.color_by==='phenotype_leiden_check_clusters' && cells.hidden.size===run.result.clusters.length-1);
  await until(()=>cells.filters?.some(filter=>filter.col===run.result.column && filter.values?.join()==='Cluster 1'));
  assert(!cells.filters.some(filter=>filter.values?.length===0), 'The image filter follows the new cluster selection rather than keeping an empty cell selection');
  assert(!cells.hidden.has('Cluster 1') && heat().querySelector('.row.picked').dataset.row==='Cluster 1','Clicking a cluster shows only its cells on the image');
  await until(()=>dataView.result?.matched===run.result.clusters.find(c=>c.name==='Cluster 1').cells && dataView.groupPick?.values.join()==='Cluster 1');
  assert(document.querySelectorAll('.data-grid tbody tr.selected').length===dataView.result.rows.length, 'Heatmap selection selects and filters the matching Data rows');
  row('Cluster 2').dispatchEvent(new MouseEvent('click',{bubbles:true,shiftKey:true}));
  await until(()=>cells.hidden.size===run.result.clusters.length-2);
  await until(()=>dataView.groupPick?.values.length===2 && dataView.result?.matched===run.result.clusters.filter(c=>['Cluster 1','Cluster 2'].includes(c.name)).reduce((n,c)=>n+c.cells,0));
  assert(tools().querySelectorAll('[data-group].picked').length===2, 'Multiple heatmap groups agree with the left list and Data');
  // Two clusters picked are named together from the bar above the heatmap.
  const figures=document.querySelector('#studio-figures');
  await until(()=>/2 picked/.test(figures.textContent));
  const together=figures.querySelector('[aria-label="Name the picked clusters"]');together.value='Mixed';
  button(figures,'Name 2').click();
  await until(()=>tools().querySelector('[aria-label="Phenotype of Cluster 1"]').value==='Mixed' && tools().querySelector('[aria-label="Phenotype of Cluster 2"]').value==='Mixed');
  assert(/Mixed \(1\)/.test(heat().textContent) && tools().querySelector('[aria-label="Colour of Cluster 1"]').value===tools().querySelector('[aria-label="Colour of Cluster 2"]').value,
    'Picked clusters are named together and share a colour');
  button(document.querySelector('#studio-figures'),'Show all').click();
  await until(()=>cells.hidden.size===0);
  assert(true,'Shift adds a cluster, Show all brings every cluster back');
  document.querySelector('[aria-label="Only selected cells"]').click();
  await until(()=>dataView.result?.matched===40);
  const firstRow=document.querySelector('.data-grid tbody tr');
  const values=dataView.result.shown, groupValue=dataView.result.rows[0][values.indexOf(run.result.column)+1];
  firstRow.click();
  await until(()=>tools().querySelector('[data-group="'+groupValue+'"]')?.classList.contains('picked'));
  assert(heat().querySelector('.row.picked')?.dataset.row===groupValue, 'Selecting a Data row selects its cluster in the left panel and heatmap');
  button(document.querySelector('#studio-figures'),'Show all').click();
  // The explorer: a group picked in the list shows its top markers and where it is.
  const image=()=>state.layers.find(layer=>layer.type==='image');
  const line=tools().querySelector('[data-group="Cluster 0"]');line.click();
  await until(()=>/Where: 1 ROI/.test(tools().textContent) && cells.hidden.size===run.result.clusters.length-1);
  await until(()=>image().channels.filter(c=>c.visible).length===2);
  assert(true,'Clicking a group turns its top markers on');
  const shown=image().channels.find(c=>c.visible);shown.visible=false;
  const top=run.result.clusters.find(c=>c.name==='Cluster 0').top_markers;
  assert(top.length===2 && tools().querySelector('.studio-where').textContent.includes('roi'),'A picked group lists its top markers and the ROIs it is in');
  button(tools(),'Top markers on the image').click();
  await until(()=>image().channels.filter(c=>c.visible).length===2);
  assert(true,'Its top markers are turned on on the image');
  line.click();await until(()=>cells.hidden.size===0 && !tools().querySelector('.studio-where'));
  button(tools(),'Show on the image').click();
  await until(()=>state.layers.some(layer=>layer.color_by==='phenotype_leiden_check_clusters' && layer.id.startsWith('result-')));
  assert(true,'The clusters can also be added as their own layer');
  tools().querySelectorAll('[aria-label^="Phenotype of Cluster"]').forEach((input,index)=>{input.value=index%2?'Tumor':'T cell';input.dispatchEvent(new Event('change'));});
  const swatch=tools().querySelector('[aria-label="Colour of Cluster 0"]');swatch.value='#123456';swatch.dispatchEvent(new Event('input'));
  button(tools(),'Save names').click();
  for(let i=0;i<600;i++){const found=(await jobs()).find(job=>job.kind==='name_clusters');if(found && /done|failed/.test(found.status))break;await delay(100);}
  const named=(await jobs()).find(job=>job.kind==='name_clusters');
  assert(named.status==='done' && named.result.column==='phenotype_leiden_check' && Object.keys(named.result.phenotypes).sort().join()==='T cell,Tumor','The names are saved in a new column: '+named.message);
  await until(()=>cells.color_by===named.result.column && tools().querySelector('[aria-label="Colour cells by"]').value===named.result.column);
  assert(true, 'Save names updates the cell colours and selected phenotype column');
  const kept=(await (await fetch('/spacemap-viewer/api/studio/browser-check/colours')).json()).colours;
  assert(kept.T_cell==='#123456' && /^#[0-9a-f]{6}$/.test(kept.Tumor),'Each name’s colour is saved where Tab 9 keeps it: '+JSON.stringify(kept));
  // Figures: this heatmap saved as a publication figure, and a dot plot of the named column from Publication figure.
  const panel=document.querySelector('#studio-figures');
  panel.querySelector('details.studio-options').open=true;
  button(panel,'Save figure').click();
  await until(()=>/Saved .*\.png|rror|no /.test(panel.textContent));
  assert(/Saved Leiden_clusters\.png/.test(panel.textContent),'The heatmap is saved as a publication figure: '+panel.textContent.slice(-120));
  const figureColumn=tools().querySelector('[aria-label="Figure column"]');figureColumn.dispatchEvent(new Event('focus'));
  await until(()=>Array.from(figureColumn.options).some(o=>o.value==='phenotype_leiden_check'));
  figureColumn.value='phenotype_leiden_check';
  const figureKind=tools().querySelector('[aria-label="Figure kind"]');figureKind.value='dotplot';figureKind.dispatchEvent(new Event('change'));
  await until(()=>tools().querySelector('[aria-label="Figure marker M0"]'));
  button(tools().querySelector('.studio-marker-picker:last-of-type')||tools(),'All');
  Array.from(tools().querySelectorAll('button')).filter(b=>b.textContent==='All').pop().click();
  button(tools(),'Make figure').click();
  await until(()=>/Saved phenotype_leiden_check\.png|rror/.test(tools().textContent));
  await until(()=>document.querySelector('#studio-figures img.studio-figure-preview'));
  assert(/Saved phenotype_leiden_check\.png/.test(tools().textContent),'Publication figure makes a dot plot of any phenotype column');
  // The assistant's naming plan (its reply stood in for here: no model runs in the check) lists every cluster with an
  // editable name; Do it saves them and the explorer shows them.
  const realFetch=window.fetch;
  // The answer arrives as it is written: JSON lines, the last one {done: reply}.
  let sentCluster;
  window.fetch=async(url,options)=>String(url).endsWith('/chat')&&options&&options.method==='POST'?(sentCluster=JSON.parse(options.body),new Response(JSON.stringify({text:'{"say"'})+'\n'+JSON.stringify({done:{conversation:'check',
    say:'Cluster 0: M0 high; Cluster 1: M1 high.',action:{name:'name_clusters',args:{job_id:run.id,names:{'Cluster 0':'Alpha','Cluster 1':'Beta'}}},runs_now:false,source:'model',remember:[]}})+'\n',
    {status:200,headers:{'Content-Type':'application/x-ndjson'}})):realFetch(url,options);
  tools().querySelector('[aria-label="Ask AI about Cluster 0"]').click();
  await until(()=>document.querySelector('[aria-label="Name of Cluster 1"]'));
  assert(sentCluster.page.explorer.job_id===run.id && sentCluster.page.explorer.picked.join()==='Cluster 0' && sentCluster.text.includes('full marker profiles'), 'Ask AI sends the selected run, cluster, reference column and comparison request');
  assert(tools().querySelector('[aria-label="Phenotype of Cluster 0"]').value==='Alpha' && heat().textContent.includes('Alpha'), 'The AI response fills the left name boxes and heatmap before saving');
  window.fetch=realFetch;
  document.querySelector('[aria-label="Name of Cluster 1"]').value='Beta cells';
  button(document.querySelector('#studio-assistant .studio-action'),'Do it').click();
  await until(()=>tools().querySelector('[aria-label="Phenotype of Cluster 1"]').value==='Beta cells' && tools().querySelector('[aria-label="Phenotype of Cluster 0"]').value==='Alpha');
  for(let i=0;i<600;i++){const latest=(await jobs()).find(job=>job.kind==='name_clusters');if(latest && latest.id!==named.id && /done|failed/.test(latest.status))break;await delay(100);}
  const second=(await jobs()).find(job=>job.kind==='name_clusters');
  assert(second.id!==named.id && second.status==='done' && Object.keys(second.result.phenotypes).includes('Beta cells') && second.result.column==='phenotype_leiden_check',
    'The assistant’s edited names update the same phenotype column and explorer: '+second.result.column);
  await until(()=>cells.color_by===second.result.column);
  assert(true, 'The cell layer switches to the saved phenotype names after the job succeeds');
  // Jobs > Saved results: Studio's columns wait beside the table until Write into the CSV.
  document.querySelector('[data-tab=jobs]').click();
  const jobsPane=document.querySelector('#studio-jobs');
  await until(()=>/phenotype_leiden_check/.test(jobsPane.textContent));
  assert(/phenotype_leiden_check_clusters/.test(jobsPane.textContent) && button(jobsPane,'Write into the CSV'),'Saved results list the columns waiting to be written into the CSV');
  // Re-running requires an explicit, one-run overwrite choice beside Save as.
  const overwrite=tools().querySelector('[aria-label="Overwrite existing results"]');
  assert(overwrite && !overwrite.checked, 'Overwrite is off by default');
  button(document.querySelector('.studio-runbar'),'Run Leiden').click();
  await until(()=>document.querySelector('.studio-runbar').textContent.includes('tick Overwrite existing results'));
  assert((await jobs()).filter(job=>job.kind==='leiden').length===1, 'Without the checkbox the old results are kept');
  overwrite.click();
  set('Leiden result column','phenotype_changed');
  assert(!overwrite.checked, 'Changing the output column clears the overwrite choice');
  set('Leiden result column','phenotype_leiden_check');
  overwrite.click();
  button(document.querySelector('.studio-runbar'),'Run Leiden').click();
  let rerun;
  for(let i=0;i<1800;i++){
    rerun=(await jobs()).find(job=>job.kind==='leiden' && job.id!==run.id);
    if(rerun && /done|failed/.test(rerun.status))break;
    await delay(100);
  }
  assert(rerun?.status==='done' && rerun.result.overwrite && rerun.result.column===run.result.column,
    'The checked rerun replaces the existing cluster column: '+rerun?.message);
  assert(!overwrite.checked, 'Overwrite resets after starting a run');
  await until(()=>Array.from(tools().querySelectorAll('[aria-label^="Phenotype of Cluster"]')).every(input=>!input.value));
  assert(!heat().textContent.includes('Alpha') && !heat().textContent.includes('Beta cells'), 'Previous names are cleared from the rerun’s fields and heatmap');
  const newName=tools().querySelector('[aria-label="Phenotype of Cluster 0"]');
  newName.value='Reviewed';newName.dispatchEvent(new Event('change'));
  button(tools(),'Save names').click();
  let renamed;
  for(let i=0;i<600;i++){
    renamed=(await jobs()).find(job=>job.kind==='name_clusters');
    if(renamed && renamed.id!==second.id && /done|failed/.test(renamed.status))break;
    await delay(100);
  }
  assert(renamed?.status==='done' && renamed.result.column==='phenotype_leiden_check',
    'Naming the replacement run writes into the selected output, without another numbered column');
  await until(()=>cells.color_by==='phenotype_leiden_check');
  assert(true, 'The viewer follows the replacement names');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,clusters:run.result.clusters.length})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}
},100);
</script>
"""


def setup(root):
    """A marker set saved in the image viewer's right-hand pane."""
    import json
    import pandas as pd
    import shutil

    table = pd.read_csv(root / "cells.csv")
    table["slide"] = "physical_slide_1"
    table["slide_id"] = "patient_1_T1"
    table.to_csv(root / "cells.csv", index=False)
    images = root / "phenotyping_rois"
    images.mkdir()
    shutil.copyfile(root / "roi.ome.tiff", images / "roi.ome.tiff")
    (root / (root.name + '.json')).write_text(json.dumps({"directory_mappings": {"viz_roi_dir": str(images)}}))

    folder = root / "outputs/viewer/display"
    folder.mkdir(parents=True, exist_ok=True)
    sets = {"phenotyping_markers": [{"name": "M0", "color": "#ff0000"}, {"name": "M1", "color": "#00ff00"}]}
    (folder / "marker_sets.json").write_text(json.dumps({"sets": sets}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup)
