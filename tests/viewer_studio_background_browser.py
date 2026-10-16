"""Browser-check Studio's Background step on a disposable image in headless Firefox.

Usage: PYTHONPATH=src python tests/viewer_studio_background_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<2200;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
// Headless Firefox answers no dialogs: record each question and answer no.
const confirms=[];window.confirm=text=>{confirms.push(text);return false;};
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab2]'));
  document.querySelector('[data-step=tab2]').click();
  await until(()=>document.querySelector('#studio-background [aria-label$=" level"]'));
  const image=state.layers.find(item=>item.type==='image'), viewer=state.viewer;
  const box=()=>document.querySelector('#studio-background');
  const levels=()=>Array.from(box().querySelectorAll('[aria-label$=" level"]'));
  assert(levels().length===image.channels.length,'Every marker has a level');
  assert(box().querySelector('.auto-controls [aria-label="Auto method"]'),'Auto method and region are in the step');
  levels()[0].value='55';levels()[0].dispatchEvent(new Event('change'));
  assert(image.channels[0].low===55,'A typed level is the marker minimum on the image');
  assert(Number(image.inputs[image.channels[0].index].low.value)===55,'The image card follows the Background step');
  const before=image.channels[1].low;
  Array.from(box().querySelectorAll('.level-row button')).filter(item=>item.textContent==='Auto')[1].click();
  await until(()=>!image.autoRun && image.channels[1].low!==before);
  await until(()=>Number(levels()[1].value)===Number(image.channels[1].low.toFixed(1)));
  assert(true,'Auto measures one marker and its level shows');
  const bar=document.querySelector('.studio-runbar');
  const row=text=>Array.from(bar.querySelectorAll('.row')).find(item=>item.querySelector('.lbl')?.textContent===text);
  assert(!button(document.querySelector('#studio-tools'),'Pick all'),'No Pick all button: the box on top of the tick column');
  await until(()=>document.querySelector('#studio-tools .studio-pairs th input[type=checkbox]'));
  const head=document.querySelector('#studio-tools .studio-pairs th input[type=checkbox]');
  head.click();
  assert(Array.from(document.querySelectorAll('#studio-tools .studio-pairs td input[type=checkbox]')).every(item=>item.checked),'The box on top ticks every image');
  document.querySelector('#studio-tools .studio-pairs th input[type=checkbox]').click();
  assert(Array.from(document.querySelectorAll('#studio-tools .studio-pairs td input[type=checkbox]')).every(item=>!item.checked),'and clears them');
  const lowBefore=image.channels.map(c=>c.low);
  button(row('Detect'),'This image').click();
  await until(()=>/Levels detected on/.test(bar.textContent));
  assert(image.channels.some((c,i)=>c.low!==lowBefore[i]),'Detect on this image sets the markers\u2019 minimums with the Auto method');
  levels()[0].value='55';levels()[0].dispatchEvent(new Event('change'));
  button(row('Export'),'This image').click();
  await until(()=>bar.textContent.includes('Saved 2 background levels'));
  const list=await (await fetch('/spacemap-viewer/api/studio/browser-check/step/tab2')).json();
  assert(list.items.find(item=>item.name==='roi').state_label==='Levels saved','The image shows its saved levels');
  assert(document.querySelector('#studio-action-status button')?.textContent==='Undo','Saving levels can be undone');
  await until(()=>box().textContent.includes('Same as exported') && box().querySelector('.level-saved').textContent==='55.0');
  assert(true,'Each marker shows its saved level');
  levels()[0].value='80';levels()[0].dispatchEvent(new Event('change'));
  await until(()=>button(box(),'Load exported levels') && !button(box(),'Load exported levels').hidden);
  button(box(),'Load exported levels').click();
  await until(()=>image.channels[0].low===55);
  assert(true,'Saved levels can be loaded back onto the image');
  levels()[0].value='60';levels()[0].dispatchEvent(new Event('change'));
  button(row('Export'),'This image').click();
  await until(()=>confirms.length);
  assert(confirms.length===1 && confirms[0].includes('1 marker differ'),'Replacing different saved levels asks first');
  const kept=await (await fetch('/spacemap-viewer/api/studio/browser-check/background_levels')).json();
  assert(kept.levels[image.channels[0].name]===55,'Answering no keeps the saved table');
  row('Export').querySelector('[aria-label="Export thresholds as CSV"]').click();
  row('Export').querySelector('[aria-label="Export background-removed images as OME-TIFF"]').click();
  button(row('Export'),'This image').click();
  const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  for(let i=0;i<600;i++){if((await jobs()).some(job=>job.kind==='export_background' && /done|failed/.test(job.status)))break;await delay(100);}
  const job=(await jobs()).find(item=>item.kind==='export_background');
  assert(job.status==='done','The job writes the background-removed image: '+job.message);
  assert(!document.querySelector('#studio-jobs').hidden,'The Jobs tab opens for the running job');
  document.querySelector('#studio-tools [aria-label="Refresh list"]')?.click();

  await until(()=>document.querySelector('[data-step=tab2] .studio-dot').dataset.state!=='unknown');
  assert(true,'The rail shows the step state');
  assert(state.viewer===viewer,'The image stays open throughout');
  // Before / after: the image as measured (minimums 0), or as removing the levels leaves it; the levels stay.
  const tools=document.querySelector('#studio-tools');
  const levelsBefore=JSON.stringify(image.channels.map(c=>c.low));
  button(tools,'Before').click();
  assert(image.showBefore===true && button(tools,'Before').getAttribute('aria-pressed')==='true','Before shows the markers as measured');
  assert(JSON.stringify(image.channels.map(c=>c.low))===levelsBefore,'The levels are kept while Before is shown');
  button(tools,'After').click();
  assert(image.showBefore===false && button(tools,'After').getAttribute('aria-pressed')==='true','After shows what removing the levels leaves');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}
},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180)
