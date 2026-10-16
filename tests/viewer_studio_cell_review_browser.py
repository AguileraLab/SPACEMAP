"""Marker-list ranking, shared phenotype preset and automatic trained suggestions."""
from viewer_threshold_browser import main


def setup(root):
    from spacemap.viewer import display
    from spacemap_gui.viewer import api
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    import pandas as pd
    from spacemap import dataio, phenotype_presets
    from spacemap_gui.studio import auto_reference, reference_models
    auto_reference.QUIET_SECONDS = .2
    auto_reference.MIN_INTERVAL = .5
    table = root / "cells.csv"
    original = pd.read_csv(table)
    original["cell_id"] = range(1001,1041)
    original.to_csv(table,index=False)
    dataio.ensure_fast_copy(table,force=True)
    display.save_set(display.sets_path(root/"display.json"),"phenotyping",
                     [{"name":"M0","color":"#f1a234","low":0,"high":120}])
    display.save_set(display.sets_path(root/"display.json"),"unavailable",
                     [{"name":"Absent","color":"#ffffff","low":0,"high":120}])
    async def check(request):
        return JSONResponse({"entries":phenotype_presets.read()["entries"],
            "models":reference_models.available(table),
            "live":dataio.load(table, columns=["phenotype_ref1"]).phenotype_ref1.tolist(),
            "unchanged":pd.read_csv(table).phenotype_ref1.equals(original.phenotype_ref1)})
    api.app.router.routes.append(Route("/cell-review-test",check))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change',{bubbles:true}));};
const button=(parent,text)=>Array.from(parent?.querySelectorAll('button') || []).find(b=>b.textContent===text);
const read=async()=>await(await fetch('/spacemap-viewer/api/cell-review-test')).json();
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent(event.message+' '+event.filename+':'+event.lineno)));
window.addEventListener('unhandledrejection',event=>fetch('/result?error='+encodeURIComponent(String(event.reason))));
setTimeout(async()=>{try{
  const s=window.__state;await until(()=>s.cellsReady&&s.imagesReady&&document.querySelector('[aria-label="Phenotyping method"]'));
  fetch('/result?progress=Viewer%20ready');
  change(document.querySelector('[aria-label="Phenotyping method"]'),'reference');
  await until(()=>document.querySelector('.manual-reference-card'));
  fetch('/result?progress=Manual%20controls%20ready');
  setTool('off');
  s.layerGroups.cells=true;buildPanel();
  const image=()=>s.layers.find(l=>l.type==='image'), cells=()=>s.layers.find(l=>l.type==='cells');
  const controls=()=>document.querySelector('.cell-marker-controls'), card=()=>document.querySelector('.manual-reference-card');
  const shown=()=>image().channels.filter(c=>c.visible).map(c=>c.name).join(',');
  const click=(x=55)=>s.viewer.raiseEvent('canvas-click',{quick:true,position:s.viewer.viewport.pixelFromPoint(image().item.imageToViewportCoordinates(x,15)),originalEvent:{}});
  await until(()=>controls()?.querySelector('option[value="phenotyping"]'));
  fetch('/result?progress=Marker%20sets%20ready');
  const styles=JSON.stringify(image().channels.map(c=>[c.color,c.low,c.high]));
  await until(()=>cells().combinedTable&&cells().table===cells().combinedTable);
  s.layers.filter(l=>l.type==='annotations').forEach(l=>l.visible=false);
  setTool('off');
  click();await delay(500);
  fetch('/result?progress='+encodeURIComponent('Cell click: '+JSON.stringify({tool:s.tool,table:cells().table,cell:s.inspectedCell?.result.label,profile:s.inspectedCell?.result.profile})));
  await until(()=>s.inspectedCell?.result.profile?.status==='ready');
  controls().querySelector('[aria-label="Cell top markers"]').click();
  const count=controls().querySelector('[aria-label="Number of cell top markers"]');count.value='1';count.dispatchEvent(new Event('input'));
  await until(()=>shown()==='M1');
  change(controls().querySelector('[aria-label="Cell top marker set"]'),'phenotyping');await until(()=>shown()==='M0');
  count.value='5';count.dispatchEvent(new Event('input'));
  await until(()=>controls().textContent.includes('Only 1'));
  assert(shown()==='M0','A list shorter than the requested count shows only its available measured markers');
  assert(image().activeSet===''&&cellMarkerOptions().markerSet==='phenotyping','The saved candidate list stays selected while top markers change image visibility');
  assert(JSON.stringify(image().channels.map(c=>[c.color,c.low,c.high]))===styles,'Filtering the ranking preserves marker colours and display limits');
  button(controls().querySelector('.marker-transforms'),'Z1').click();
  await until(()=>s.inspectedCell.result.profile.steps?.includes('Z1'));
  assert(button(document.querySelector('#cellcard .marker-transforms'),'Z1').getAttribute('aria-pressed')==='true'&&shown()==='M0','Layer normalization and the cell feature panel use the same values');
  button(document.querySelector('#cellcard .marker-transforms'),'Raw').click();
  await until(()=>button(controls().querySelector('.marker-transforms'),'Raw').getAttribute('aria-pressed')==='true');
  assert(true,'Changing the cell feature transformation updates the layer controls too');
  change(controls().querySelector('[aria-label="Cell top marker set"]'),'unavailable');
  await until(()=>controls().textContent.includes('No measured markers'));
  assert(shown()==='M0','An unavailable marker list reports its limitation without falling back to other markers');
  change(controls().querySelector('[aria-label="Cell top marker set"]'),'phenotyping');
  change(document.querySelector('[aria-label="Phenotyping method"]'),'reference');
  await until(()=>card()?.querySelector('datalist option[value="Treg"]'));
  const choices=card().querySelector('[aria-label="Phenotype name to assign"]');choices.click();
  assert(card().querySelectorAll('.reference-choice-menu .swatch').length>=12,'The phenotype dropdown includes the twelve preset names with colours');
  card().querySelector('[aria-label="Edit phenotype preset"]').click();
  await until(()=>document.querySelector('.phenotype-preset-rows')?.children.length===12);
  const dialog=document.querySelector('.phenotype-preset-editor');
  button(dialog,'Add phenotype').click();
  const added=dialog.querySelector('.phenotype-preset-rows').lastElementChild;
  added.querySelector('input:not([type=color])').value='My phenotype';added.querySelector('input[type=color]').value='#123456';
  button(dialog,'Save preset').click();
  await until(()=>!document.querySelector('.phenotype-preset-editor'));
  await until(()=>card()?.querySelector('datalist option[value="My phenotype"]'));
  assert((await read()).entries.some(item=>item.name==='My phenotype'&&item.color==='#123456')&&(await read()).unchanged,'Custom names and colours persist without changing any cell labels');
  const models=card().querySelector('.reference-model-training');models.open=true;
  await until(()=>button(card(),'Train from this reference'));
  cells().suggestionsOn=false;cells().suggestionMode='saved_reference';cells().suggestionModelId=null;
  button(card(),'Train from this reference').click();
  models.open=false;
  await until(()=>document.querySelector('.cell-suggestions .suggestion-choice'));
  assert(cells().suggestionsOn&&cells().suggestionMode==='saved_reference'&&!!cells().suggestionModelId,'A completed training run enables its saved model and immediately suggests on the selected cell');
  const originalFetch=window.fetch;let emptyRequests=0;
  window.fetch=(url,args)=>{if(String(url).endsWith('/suggestions')){const payload=JSON.parse(args.body);if(payload.mode==='saved_reference'&&!payload.model_id)emptyRequests++;}return originalFetch(url,args);};
  cells().suggestionModelId=null;click(95);
  await until(()=>s.inspectedCell?.result.label===3&&s.inspectedCell.result.suggestions?.status==='ready');
  assert(emptyRequests===0&&!!cells().suggestionModelId,'A missing saved model selection recovers before making a prediction request');
  assert((await read()).unchanged,'Training and predictions do not silently assign phenotypes');
  card().querySelector('[aria-label="Auto-train suggestions"]').click();
  await until(()=>s.autoReference?.enabled&&s.autoReference.completed?.result?.auto_train);
  const firstAuto=s.autoReference.completed.id;
  const firstModel=s.autoReference.completed.result.models[0].model_id;
  await until(()=>s.inspectedCell?.result.suggestions?.provenance?.auto_train);
  assert(s.autoReference.column==='phenotype_ref1','Auto-train learns the manual reference and supplies suggestions without a manual training click');
  const labelName=card().querySelector('[aria-label="Phenotype name to assign"]');
  labelName.value='B';labelName.dispatchEvent(new Event('input'));
  await until(()=>button(card(),'Assign 1 cells')&&!button(card(),'Assign 1 cells').disabled);
  button(card(),'Assign 1 cells').click();
  await until(async()=>(await read()).live[2]==='B');
  await until(()=>s.autoReference?.completed?.id!==firstAuto&&s.autoReference.completed?.result?.auto_train);
  assert(true,'A saved manual assignment triggers one newer background model');
  const savedModels=(await read()).models;
  assert(savedModels.filter(m=>m.auto_train).length===1&&savedModels.filter(m=>!m.auto_train).length===2,
    'Auto updates replace the older model and keep manually trained models');
  const latest=savedModels.find(m=>m.auto_train);
  cells().suggestionModelId=firstModel;buildPanel();
  await until(()=>cells().suggestionModelId===latest.model_id);
  assert(latest.replaces.includes(firstModel),'An older saved selection follows its replacement in the model menu');
  // buildPanel recreates the controls; wait for their saved Auto-train state.
  await until(()=>card()?.querySelector('[aria-label="Auto-train suggestions"]')?.checked);
  card().querySelector('[aria-label="Auto-train suggestions"]').click();
  await until(()=>s.autoReference?.enabled===false);
  assert(!card().querySelector('[aria-label="Auto-train suggestions"]').checked,'Auto-train can be switched off while keeping the saved model');
  button(card(),'Undo').click();
  await until(async()=>(await read()).live[2]==='A');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#cellcard')?.textContent+'\n'+document.querySelector('.manual-reference-card')?.textContent));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER,studio=True,timeout=240,setup=setup,screenshot=False)
