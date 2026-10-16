"""Synthetic browser check for Live RESOLVE, shared options and the job result."""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<2200;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>document.querySelector('[aria-label="Phenotyping method"]'));
  const picker=document.querySelector('[aria-label="Phenotyping method"]');picker.value='resolve';picker.dispatchEvent(new Event('change'));
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('#studio-resolve button'));
  fetch('/result?progress=image-and-cells-ready');
  const viewer=state.viewer, image=state.layers.find(item=>item.type==='image'), cells=state.layers.find(item=>item.type==='cells');
  const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
  const left=document.querySelector('#studio-resolve');
  assert(button(left,'Start live RESOLVE'),'Live RESOLVE is in the Phenotyping step');
  button(left,'Start live RESOLVE').click();await until(()=>cells.live);
  fetch('/result?progress=live-resolve-started');
  assert(left.textContent.includes('40 cells'),'The live result shows the correct number of cells');
  assert(document.querySelectorAll('.live-resolve').length===1,'Studio does not duplicate Live RESOLVE controls');
  await until(()=>left.querySelector('[aria-label="'+cells.live.markers[0].marker+' threshold"]'));
  const firstMarker=cells.live.markers[0], channel=image.channels.find(c=>c.name===firstMarker.marker);
  const thresholdInput=left.querySelector('[aria-label="'+firstMarker.marker+' threshold"]');
  thresholdInput.value=String(firstMarker.threshold+3);thresholdInput.dispatchEvent(new Event('change'));
  assert(!channel || channel.low===firstMarker.threshold,'A RESOLVE threshold is the image marker\'s minimum');
  await until(()=>!cells.live.busy && !cells.live.pending);
  assert(document.querySelector('#studio-tools .studio-work-row'),'The step has a work list');
  const options=document.querySelector('.studio-method .studio-options');options.open=true;
  await until(()=>options.querySelector('[aria-label="Classic output column"]'));
  const output=options.querySelector('[aria-label="Classic output column"]');output.value='phenotype_shared_test';
  button(options,'Save shared options').click();
  await until(()=>options.textContent.includes('Saved for Studio'));
  fetch('/result?progress=settings-saved');
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/settings')).json();
  assert(saved.values.output_col==='phenotype_shared_test','Studio saves the classic RESOLVE setting');
  const newColumn=document.querySelector('.studio-runbar [aria-label="New result column"]');newColumn.value='phenotype_job_test';
  button(document.querySelector('.studio-runbar'),'Run all ROIs').click();
  fetch('/result?progress=job-requested');
  await until(()=>document.querySelector('#studio-jobs').textContent.includes('Show result'));
  fetch('/result?progress=job-finished');
  assert(state.viewer===viewer,'The image stays open while the job runs');
  const jobState=await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json();
  assert(jobState.jobs[0].status==='done' && jobState.jobs[0].result.cells===40,'The background job completes with every cell');
  assert(jobState.jobs[0].usage.duration_s>0,'The job records its resource use');
  button(document.querySelector('#studio-jobs'),'Show result').click();
  await until(()=>state.layers.some(item=>item.color_by==='phenotype_job_test'));
  fetch('/result?progress=result-layer-shown');
  assert(state.viewer===viewer,'The saved result becomes a layer without reopening the image');
  const search=document.querySelector('[aria-label="Find a step or image"]');search.value='count cells';
  button(document.querySelector('#studio-header'),'Ask').click();
  await until(()=>document.querySelector('#studio-action-status').textContent.includes('40 cells counted'));
  assert(document.querySelector('.studio-counts'),'Cell counts are returned beside the view');
  assert(state.viewer===viewer && state.layers.includes(image),'Counting preserves the image');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,job:jobState.jobs[0].status})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-steps')?.textContent));}
},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180)
