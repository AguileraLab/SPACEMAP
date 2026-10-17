"""Ctrl-click inspection averages transformed markers and complete model scores."""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.cellsReady);
 const image=s.layers.find(l=>l.type==='image'),layer=s.layers.find(l=>l.id==='cells');
 const click=(label,ctrl=false)=>{const n=label-1;s.viewer.raiseEvent('canvas-click',{quick:true,
   position:s.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(40*(n%8)+15,40*Math.floor(n/8)+15)),originalEvent:{ctrlKey:ctrl}});};
 const result=()=>s.inspectedCell?.result,ready=()=>result()?.profile?.status==='ready';
 s.layerGroups.cells=true;buildPanel();
 document.querySelector('[aria-label="Cell top markers"]').click();
 const number=document.querySelector('[aria-label="Number of cell top markers"]');number.value=1;number.dispatchEvent(new Event('input'));
 click(1);await until(()=>ready()&&result().label===1);click(21,true);
 await until(()=>ready()&&result().members?.length===2);
 assert(document.querySelector('#cellcard h4').textContent.includes('2 cells selected'),'Ctrl-click opens one card for the selected cells');
 assert(result().details.phenotype_ml_consensus==='A (1/2), B (1/2)','Mixed saved predictions remain visible with counts');
 assert(result().profile.markers.find(m=>m.marker==='M0').mean===101,'Raw markers use the arithmetic mean across selected cells');
 await until(()=>image.channels.filter(c=>c.visible).map(c=>c.name).join(',')==='M1');
 assert(true,'Top markers uses the selection average');
 await until(()=>s.cellOutline?.length===2);assert(true,'Both selected cell outlines are highlighted');
 const transform=name=>Array.from(document.querySelectorAll('#cellcard .marker-transforms button')).find(b=>b.textContent===name);
 transform('Arcsinh').click();await until(()=>ready()&&result().profile.steps?.includes('Arcsinh'));
 const actual=result().profile.markers.find(m=>m.marker==='M0').view;
 assert(Math.abs(actual-(Math.asinh(1/5)+Math.asinh(201/5))/2)<1e-10,'Transform each cell before averaging, rather than transforming the mean');
 for(const step of ['Z1','Z2','CDF','NLog']){transform(step).click();await until(()=>ready()&&result().profile.steps.includes(step));}
 const memberValues=result().members.map(m=>m.profile.markers.find(r=>r.marker==='M1').view);
 assert(Math.abs(result().profile.markers.find(m=>m.marker==='M1').view-(memberValues[0]+memberValues[1])/2)<1e-10,'Combined transformations also average individual cell values');
 layer.suggestionsOn=true;layer.suggestionMode='consensus_rf';
 await requestSuggestions(layer,result(),s.cardSeq);
 await until(()=>result()?.suggestions?.status==='ready');
 const selected=result(), settings=suggestionSettings(layer);
 const individual=await Promise.all([1,21].map(label=>postJSON('/cells/'+layer.token+'/suggestions',{...settings,label})));
 const scores=new Map();individual.forEach(v=>v.probabilities.forEach(c=>scores.set(c.phenotype,(scores.get(c.phenotype)||0)+c.probability/2)));
 assert(selected.suggestions.scored===2&&selected.suggestions.choices.every(c=>Math.abs(c.probability-scores.get(c.phenotype))<1e-10),'Selected-cell suggestions average the full individual model distributions');
 assert(document.querySelector('#cellcard').textContent.includes('2/2 cells scored'),'Prediction coverage is shown for the group');
 const accepted=selected.suggestions.choices[0].phenotype;
 document.querySelector('#cellcard .suggestion-choice').click();
 await until(()=>!window.__labelSaves.busy&&!window.__labelSaves.queue.length);
 const labelled=await Promise.all([1,21].map(label=>getJSON('/cells/'+layer.token+'/at?label='+label+'&details_only=1&t='+encodeURIComponent(layer.table))));
 assert(labelled.every(value=>value.details[s.labelColumn]===accepted),'Assign all explicitly saves the suggested label for every selected cell');
 layer.suggestionsOn=false;click(21,true);await until(()=>ready()&&result().label===1&&!result().members);
 assert(true,'Ctrl-click on a selected cell removes it');
 click(5,true);click(6,true);await until(()=>ready()&&result().members?.length===3);
 assert(result().labels.join(',')==='1,5,6','Rapid Ctrl-clicks retain every selected cell');
 click(2);await until(()=>ready()&&result().label===2&&!result().members);
 assert(s.cellInspectionSelection.members.size===1,'Normal clicks reset the group');
 click(2,true);await until(()=>!s.inspectedCell);
 assert(document.querySelector('#cellcard').hidden&&!s.cellOutline,'Removing the final selected cell clears its card and outline');
 click(1);await until(()=>ready()&&result().label===1);
 const originalFetch=window.fetch;let waiting=false,release;
 window.fetch=async(url,args)=>{const response=await originalFetch(url,args);if(String(url).endsWith('/profile')&&JSON.parse(args.body).label===21){waiting=true;await new Promise(r=>release=r);}return response;};
 click(21,true);await until(()=>waiting);click(3);await until(()=>ready()&&result().label===3);
 const visibility=image.channels.map(c=>c.visible).join(',');release();await delay(200);
 assert(result().label===3&&image.channels.map(c=>c.visible).join(',')===visibility,'A stale group response cannot replace a newer single-cell inspection');
 window.fetch=originalFetch;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#cellcard')?.textContent));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, timeout=180)
