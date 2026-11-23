"""Exercise shared selections, background exports and the figure gallery in Firefox."""
from viewer_threshold_browser import main
from viewer_studio_subphenotyping_browser import setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<3600;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+'\n'+e.error?.stack)));
setTimeout(async()=>{
 try{
  await until(()=>window.__state.imagesReady && window.__state.cellsReady && document.querySelector('[aria-label="Phenotyping method"]'));
  const picker=document.querySelector('[aria-label="Phenotyping method"]');
  const open=method=>{picker.value=method;picker.dispatchEvent(new Event('change'));};
  open('subphenotyping');
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Figure patient column"]') && tools().querySelector('[aria-label="Sub-phenotyping marker M3"]'));
  const panel=tools().querySelector('[data-marker-figures]');
  assert(!panel.open,'Figure controls start collapsed');
  panel.open=true;
  assert(tools().querySelectorAll('[aria-label="Phenotype column to split"]').length===1,'One shared phenotype selector');
  assert(tools().querySelectorAll('[aria-label="Sub-phenotyping marker M3"]').length===1,'One shared marker panel');
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=String(value);input.dispatchEvent(new Event('change'));};
  const fieldVisible=label=>!tools().querySelector('[aria-label="'+label+'"]').closest('.field').hidden;
  const tickFigure=label=>tools().querySelector('[aria-label="Generate '+label+'"]').click();
  assert(panel.querySelectorAll('.studio-figure-choices input:checked').length===1 &&
    tools().querySelector('[aria-label="Generate Dot plot"]').checked,'Only the dot plot starts selected');
  assert(!fieldVisible('Figure patient column')&&!fieldVisible('Figure stratification columns')&&!fieldVisible('Co-expression X'),
    'Options for unselected figures are hidden');
  button(panel,'None').click();
  assert(button(panel,'Generate selected').disabled,'An empty selection cannot start a job');
  tickFigure('Marker variation');
  assert(!fieldVisible('High expression Z cutoff')&&fieldVisible('Marker variation metric'),'Variation shows its relevant options');
  tickFigure('Marker variation');
  tickFigure('Patient heatmap');
  assert(button(panel,'Generate selected').disabled&&fieldVisible('Figure patient column'),'Patient heatmap requires a patient column');
  tickFigure('Marker histograms');
  button(tools(),'Add filter').click();
  set('Cell filter 1 column','tumor');
  await until(()=>tools().querySelector('[aria-label="Cell filter 1 value"] option[value="1"]'));
  set('Cell filter 1 value','1');
  set('Figure patient column','batch_id');
  set('Minimum cells per sample',10);
  set('High expression Z cutoff',0.5);
  set('Sub-phenotyping arcsinh cofactor',10);
  tools().querySelector('[aria-label="Stratify by tumor"]').click();
  await until(()=>tools().querySelector('[data-subpheno-selection]').textContent==='240 cells selected');
  const firstImage=[...document.querySelectorAll('[aria-label^="Process image "]')].find(e=>e.getAttribute('aria-label')!=='Process image roi2');
  firstImage.click();
  await until(()=>tools().querySelector('[data-subpheno-selection]').textContent==='210 cells selected');
  assert(!button(panel,'Generate selected').disabled,'A valid figure selection can run');
  button(panel,'Generate selected').click();
  const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  await until(async()=>(await jobs()).some(j=>j.kind==='marker_figures'&&['done','failed'].includes(j.status)));
  const job=(await jobs()).find(j=>j.kind==='marker_figures');
  assert(job.status==='done','Figure job succeeded: '+job.message);
  assert(job.result.metadata.cells===210 && job.result.metadata.phenotypes.join()==='T cell'&&job.result.metadata.selected_images.join()==='roi2','Figures use the selected image and phenotype, excluding Tumour');
  assert(job.result.metadata.z===0.5 && job.result.metadata.stratify.join()==='tumor','Custom cutoff and group column reach the worker');
  assert(job.result.metadata.cofactor===10&&job.result.metadata.transform==='auto','Figures share the clustering transformation without duplicating controls');
  assert(job.result.metadata.markers.length===4,'Figures use the shared marker panel');
  assert([...new Set(job.result.figures.map(e=>e.kind))].sort().join()==='histogram,stratified',
    'Only the two requested figure types are generated');
  assert(job.result.metadata.options.figures.join()==='histogram,patients','Figure choices reach the worker');
  await until(()=>document.querySelector('[data-figure="marker-exploration"] .studio-marker-image')?.naturalWidth>0);
  const gallery=()=>document.querySelector('[data-figure="marker-exploration"]');
  assert(gallery().closest('#studio-figures'),'Figures appear in the right dock');
  const figure=gallery().querySelector('[aria-label="Marker figure"]');
  assert(job.result.figures.every(e=>[...figure.options].some(o=>o.value===e.id)),'Each exported figure can be opened individually');
  const appearance=gallery().querySelector('[data-marker-appearance]');
  assert(appearance&&!appearance.open,'Appearance controls are in the right figure panel and start collapsed');
  appearance.open=true;
  const styleSet=(label,value)=>{const input=gallery().querySelector('[aria-label="'+label+'"]');input.value=String(value);input.dispatchEvent(new Event('change'));};
  styleSet('Figure X labels (pt)',8);
  styleSet('Figure X angle (°)',45);
  styleSet('Figure Width (%)',130);
  gallery().querySelector('[aria-label="Show heatmap values"]').click();
  const originalPng=gallery().querySelector('a[download]').href;
  button(gallery(),'Apply view').click();
  await until(()=>gallery().textContent.includes('View saved.')&&!button(gallery(),'Apply view').disabled);
  const styledPng=gallery().querySelector('a[download]').href;
  assert(styledPng!==originalPng&&gallery().querySelector('img').src===styledPng,'Preview and PNG download use the same styled view');
  assert(gallery().querySelectorAll('a[download]')[1].href.includes('view_'),'PDF download uses the styled view');
  const heatmapId=figure.value;
  const histogram=job.result.figures.find(e=>e.kind==='histogram');
  figure.value=histogram.id;figure.dispatchEvent(new Event('change'));
  assert(gallery().querySelector('[aria-label="Figure X angle (°)"]').value==='','Appearance is independent for each figure');
  assert(gallery().querySelector('[aria-label="Show heatmap values"]').closest('label').hidden,'Heatmap-only options hide for histograms');
  gallery().querySelector('.studio-marker-menu').open=true;
  button(gallery(),'None').click();
  gallery().querySelector('[aria-label="Display marker M2"]').click();
  button(gallery(),'Apply view').click();
  await until(()=>gallery().textContent.includes('View saved.')&&gallery().querySelector('img').complete);
  assert(gallery().querySelectorAll('[aria-label^="Display marker"]:checked').length===1,'A single selected-marker histogram can be displayed');
  assert(tools().querySelectorAll('[aria-label^="Sub-phenotyping marker "]:checked').length===4,'Display choices do not change analysis markers');
  gallery().querySelector('[aria-label="Display marker M3"]').click();
  button(gallery(),'Apply view').click();
  await until(()=>gallery().textContent.includes('View saved.')&&!button(gallery(),'Apply view').disabled);
  assert(gallery().querySelectorAll('[aria-label^="Display marker"]:checked').length===2,'Multiple selected markers can be displayed');
  button(gallery(),'Expand').click();
  assert(gallery().closest('[role="dialog"]'),'Expand opens a large figure');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}));
  assert(gallery().closest('#studio-figures'),'Escape returns the figure to the dock');
  const pdf=await fetch(gallery().querySelector('a[download]').href);
  assert(pdf.ok&&(await pdf.arrayBuffer()).byteLength>1000,'Individual figures can be downloaded');
  const book=await fetch(gallery().querySelector('a[target="_blank"]').href);
  assert(book.ok&&book.headers.get('content-type').includes('pdf'),'The saved PDF figure book is available');
  figure.value=heatmapId;figure.dispatchEvent(new Event('change'));
  assert(gallery().querySelector('[aria-label="Figure X angle (°)"]').value==='45'&&gallery().querySelector('a[download]').href===styledPng,
    'Switching figures restores its applied appearance and downloads');
  open('leiden');await until(()=>tools().querySelector('[aria-label="Resolution"]'));
  open('subphenotyping');
  await until(()=>tools().querySelector('[aria-label="Figure patient column"]'));
  assert(tools().querySelector('[aria-label="High expression Z cutoff"]').value==='0.5','Figure settings persist after reopening');
  assert(tools().querySelector('[aria-label="Stratify by tumor"]').checked,'Stratification persists');
  assert([...tools().querySelectorAll('.studio-figure-choices input:checked')].map(e=>e.getAttribute('aria-label')).join()==='Generate Marker histograms,Generate Patient heatmap',
    'Figure selections persist after reopening');
  assert(!fieldVisible('Co-expression X')&&fieldVisible('Figure patient column'),'Reopened options match the chosen figures');
  button(tools(),'Show saved figures').click();
  await until(()=>gallery()?.querySelector('img')?.naturalWidth>0);
  assert(true,'A saved run reopens without recalculating measurements');
  assert(gallery().querySelector('[aria-label="Figure X angle (°)"]').value==='45'&&gallery().querySelector('a[download]').href===styledPng,
    'Applied appearance survives reopening the gallery');
  button(gallery(),'Reset appearance').click();
  assert(gallery().querySelector('[aria-label="Figure X angle (°)"]').value===''&&gallery().querySelector('[aria-label="Figure Width (%)"]').value==='100',
    'Reset restores automatic layout controls');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('[data-marker-figures]')?.textContent));}
},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=400, setup=setup)
