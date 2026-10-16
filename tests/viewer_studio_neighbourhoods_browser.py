"""Browser-check Studio's neighbourhoods (Spatial analysis) on a disposable project in headless Firefox.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_neighbourhoods_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<3600;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[data-step=tab8]'));
  document.querySelector('[data-step=tab8]').click();
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Phenotype column"]'));
  await until(()=>tools().querySelector('tr[data-roi="roi"]'));
  assert(/Not run/.test(tools().querySelector('tr[data-roi="roi"]').textContent)&&tools().querySelector('[aria-label="Select all"]'),'Images are listed with a tick box each and their state');
  assert(Array.from(tools().querySelectorAll('.studio-stage-tabs button')).map(b=>b.textContent).join()==='Neighbourhoods,TLS candidates,Cell–cell interactions,Density,Triads,Feature screening','The methods are tabs');
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=String(value);input.dispatchEvent(new Event('change'));};
  set('Phenotype column','phenotype_ml_consensus');set('Neighbours',5);set('Neighbourhoods',2);set('Smallest region',5);
  set('Region column','');set('Neighbourhood result column','neighborhood');
  assert(tools().textContent.includes('Not saved'),'Changed settings show as not saved');
  assert(/Find:/.test(document.querySelector('.studio-runbar').textContent),'The bottom bar reads Find: This image | Selected');
  button(document.querySelector('.studio-runbar'),'This image').click();
  try{await until(()=>tools().querySelector('[aria-label="Name of CN 0"]') || /failed/.test(document.querySelector('#studio-jobs').textContent));}
  catch(e){throw Error('No neighbourhoods; bottom bar: '+document.querySelector('.studio-runbar').textContent+' · image '+state.layers.find(l=>l.type==='image')?.name);}
  const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  const run=(await jobs()).find(job=>job.kind==='neighbourhoods');
  assert(run.status==='done' && run.result.column==='neighborhood_cn' && run.result.clusters.length===2,'Neighbourhoods run as a job and are saved: '+run.message);
  assert(JSON.stringify(run.result.images)==='["roi"]','This image runs on the open image\'s cells only');
  await until(()=>/Neighbourhoods/.test(tools().querySelector('tr[data-roi="roi"]')?.textContent||''));
  assert(true,'The list marks the image as covered by a neighbourhood run');
  const tops=run.result.clusters.map(item=>item.top_markers[0]).sort().join();
  assert(tops==='A,B','One neighbourhood is mostly A, the other mostly B: '+tops);
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/neighbourhood_settings')).json();
  assert(saved.values.k_neighbors===5 && saved.values.phenotype_col==='phenotype_ml_consensus','The settings are saved in Tab 8\'s keys');
  await until(()=>document.querySelector('#studio-figures .studio-heatmap .row'));
  const heat=document.querySelector('#studio-figures .studio-heatmap');
  assert(Array.from(heat.querySelectorAll('.column')).map(c=>c.textContent).sort().join()==='A,B','The neighbourhood heatmap has a column per phenotype');
  heat.querySelector('.row[data-row="CN 0"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
  const cells=state.layers.find(layer=>layer.type==='cells');
  await until(()=>cells.color_by==='neighborhood_cn' && cells.hidden.has('CN 1') && !cells.hidden.has('CN 0'));
  assert(true,'Clicking a neighbourhood shows only its cells on the image');
  const base='/spacemap-viewer/api/cells/'+cells.token;
  const params='?t='+encodeURIComponent(cells.table)+'&col=neighborhood_cn';
  const cellLegend=await (await fetch(base+'/legend'+params)).json();
  const mapLegend=await (await fetch(base+'/density/groups'+params)).json();
  for(const item of cellLegend.items){
    const swatch=tools().querySelector('[aria-label="Colour of '+item.value+'"]');
    const key=document.querySelector('#studio-figures .studio-heatmap [data-row="'+item.value+'"] rect[width="10"]');
    assert(swatch.value===item.color && key.getAttribute('fill')===item.color && mapLegend.items.find(v=>v.value===item.value).color===item.color,'CN colour agrees between list, heatmap, cell LUT legend and density groups: '+item.value);
  }
  const display=document.querySelector('[aria-label="Neighbourhood heatmap values"]');
  assert(display?.value==='share' && document.querySelector('#studio-figures .scale-label').textContent==='Share (%)','Neighbourhood heatmap defaults to actual percentages');
  const checkSummaries=mode=>run.result.profile.rows.every((name,r)=>{
    const matrix=mode==='share'?run.result.profile.raw:run.result.profile.values;
    const top=run.result.profile.columns.map((name,c)=>({name,value:matrix[r][c]})).filter(v=>v.value>0).sort((a,b)=>b.value-a.value||a.name.localeCompare(b.name)).slice(0,3);
    const expected=top.map(v=>v.name+' '+(mode==='share'?(100*v.value).toFixed(1)+'%':'+'+v.value.toFixed(2))).join(', ')||'No above-average share';
    return tools().querySelector('[data-group="'+name+'"] [data-summary]').textContent===expected;
  });
  assert(checkSummaries('share'),'Left summaries rank the percentages used by the heatmap');
  display.value='relative';display.dispatchEvent(new Event('change'));
  assert(checkSummaries('relative') && document.querySelector('#studio-figures .scale-label').textContent==='Z-score','Relative view synchronizes ranking and labels');
  const tooltip=document.querySelector('#studio-figures .studio-heatmap .row rect:not(.band) title');
  assert(tooltip.textContent.includes('mean share') && tooltip.textContent.includes('relative z-score'),'Tooltip gives both percentage and relative score');
  display.value='share';display.dispatchEvent(new Event('change'));
  assert(cells.hidden.has('CN 1') && !cells.hidden.has('CN 0'),'Display mode changes preserve the selected CN');
  let exported;
  const originalFetch=window.fetch;
  window.fetch=(url,options)=>{if(String(url).endsWith('/figure')&&options?.body)exported=JSON.parse(options.body);return originalFetch(url,options);};
  const figureBox=document.querySelector('[data-figure="neighbourhoods"]');
  figureBox.querySelector('details.studio-options').open=true;
  button(figureBox,'Save figure').click();
  await until(()=>!!exported);
  assert(exported.options.vmin===0 && exported.options.vmax===100 && exported.options.value_label==='Share (%)' &&
    exported.matrix.values.every((row,r)=>row.every((v,c)=>Math.abs(v-run.result.profile.raw[r][c]*100)<1e-7)),'Export uses the displayed percentage matrix and scale');
  window.fetch=originalFetch;
  await until(()=>!button(figureBox,'Save figure').disabled);
  assert(!/Suggest from/.test(tools().textContent) && !button(tools(),'Top markers on the image'),'Neighbourhoods (made of phenotypes) get no marker suggestions or top markers');
  // Several picked (Shift in the list), named together from the bar above the heatmap: one name, one colour.
  tools().querySelector('[data-group="CN 1"]').dispatchEvent(new MouseEvent('click',{bubbles:true,shiftKey:true}));
  await until(()=>document.querySelector('[aria-label="Name the picked clusters"]') && /2 picked/.test(document.querySelector('.studio-namer').textContent));
  await until(()=>!cells.hidden.has('CN 0') && !cells.hidden.has('CN 1'));
  const together=document.querySelector('[aria-label="Name the picked clusters"]');together.value='Tissue';
  button(document.querySelector('.studio-namer'),'Name 2').click();
  const nameOf=group=>tools().querySelector('[aria-label="Name of '+group+'"]').value, colourOf=group=>tools().querySelector('[aria-label="Colour of '+group+'"]').value;
  await until(()=>nameOf('CN 0')==='Tissue' && nameOf('CN 1')==='Tissue' && colourOf('CN 0')===colourOf('CN 1'));
  assert(nameOf('CN 0')==='Tissue' && nameOf('CN 1')==='Tissue' && colourOf('CN 0')===colourOf('CN 1'),'Picked neighbourhoods are named together and share a colour');
  button(tools(),'Show on the image').click();
  await until(()=>state.layers.some(layer=>layer.color_by==='neighborhood_cn' && layer.id.startsWith('result-')));
  tools().querySelectorAll('[aria-label^="Name of CN"]').forEach((input,index)=>{input.value=index?'Lower':'Upper';});
  button(tools(),'Save names').click();
  for(let i=0;i<600;i++){const found=(await jobs()).find(job=>job.kind==='name_clusters');if(found && /done|failed/.test(found.status))break;await delay(100);}
  const named=(await jobs()).find(job=>job.kind==='name_clusters');
  assert(named.status==='done' && named.result.column==='neighborhood','The names are saved in a new column: '+named.message);
  await until(()=>!button(tools(),'Save names').disabled);
  // A second edit stays in the chosen output; reruns require explicit overwrite.
  tools().querySelector('[aria-label="Name of CN 0"]').value='Updated zone';
  button(tools(),'Save names').click();
  await until(()=>!button(tools(),'Save names').disabled);
  let renamed;
  for(let i=0;i<600;i++){renamed=(await jobs()).find(job=>job.kind==='name_clusters' && job.id!==named.id);if(renamed && /done|failed/.test(renamed.status))break;await delay(100);}
  assert(renamed?.status==='done' && renamed.result.column==='neighborhood','Repeat name edits keep the chosen column');
  const overwrite=()=>tools().querySelector('[aria-label="Overwrite existing results"]');
  assert(overwrite() && !overwrite().checked,'Overwrite is visible and off by default');
  button(document.querySelector('.studio-runbar'),'This image').click();
  await until(()=>/That output column already exists/.test(document.querySelector('.studio-runbar').textContent));
  assert((await jobs()).filter(job=>job.kind==='neighbourhoods').length===1,'Unchecked overwrite prevents a replacement job');
  overwrite().checked=true;
  set('Neighbourhood result column','neighborhood_new');
  assert(!overwrite().checked,'Changing the output name clears overwrite');
  set('Neighbourhood result column','neighborhood');
  overwrite().checked=true;overwrite().dispatchEvent(new Event('change'));
  button(document.querySelector('.studio-runbar'),'This image').click();
  let rerun;
  for(let i=0;i<1200;i++){rerun=(await jobs()).find(job=>job.kind==='neighbourhoods' && job.id!==run.id);if(rerun && /done|failed/.test(rerun.status))break;await delay(100);}
  assert(rerun?.status==='done' && rerun.result.overwrite && rerun.result.columns.includes('neighborhood') && rerun.result.columns.includes('neighborhood_cn'),'Overwrite replaces groups and resets names together: '+rerun?.message);
  assert(!overwrite().checked,'Overwrite clears after the job is accepted');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}
},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400)
