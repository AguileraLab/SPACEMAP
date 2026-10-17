"""Detect/save and sequential artifact subtraction through the actual Studio controls."""
from viewer_threshold_browser import main
from viewer_artifact_review_browser import setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools'),footer=()=>document.querySelector('.studio-runbar');
 const read=p=>fetch(api(p)).then(r=>r.json()),send=(p,v)=>postJSON(p,v);
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 const stage=name=>button(document.querySelector('.studio-stage-tabs'),name).click();
 const area=features=>features.reduce((sum,f)=>{const g=f.geometry,polys=g.type==='Polygon'?[g.coordinates]:g.coordinates;return sum+polys.reduce((total,rings)=>total+rings.reduce((a,ring,i)=>a+(i?-1:1)*Math.abs(ring.reduce((v,p,j)=>{const q=ring[(j+1)%ring.length];return v+p[0]*q[1]-p[1]*q[0];},0)/2),0),0);},0);
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.find(l=>l.type==='image')?.name==='markers.ome.tiff'&&tissue()?.revision);stage('Tissue');
 await until(()=>tools().querySelector('[aria-label="Tissue label"]')&&tools().querySelector('[aria-label="Tissue annotation"]'));
 assert(!button(tools(),'Test on this image')&&!tools().textContent.includes('Test on the image'),'The separate Tissue test panel is removed');
 assert(!tools().querySelector('[aria-label="Tissue to edit"]')&&!s.layers.some(l=>l.id==='clean-tissue-regions'),'There is one working Tissue layer');
 const label=tools().querySelector('[aria-label="Tissue label"]');label.value='Review tissue';label.dispatchEvent(new Event('change'));
 await until(()=>tissue()?.revision&&!tissue().loadingShapes);  // the label change reloads the layer
 const old=tissue().revision;
 button(footer(),'This image').click();
 await until(()=>tissue()&&tissue().revision!==old&&!tissue().brushBusy&&footer().textContent.includes('Saved '));
 const detected=await read('/shapes/'+tissue().token),detectedArea=area(detected.features),file=tissue().saved_file;
 assert(detected.features.length>0&&!tissue().preview&&detected.features.every(f=>f.properties.name.startsWith('Review tissue_')),'This image uses Label and saves detected polygons immediately');
 const preview=await send('/studio/browser-check/tissue_preview',{});
 assert(area(preview.features)===detectedArea,'The saved detection matches the former test calculation');
 stage('Artifacts');await until(()=>tools().querySelector('[aria-label="Work within"]')&&s.layers.find(l=>l.id==='artifact-clot')?.revision);
 assert(tools().querySelector('[aria-label="Work within"]').value==='tissue','The detected Tissue is selected for artifacts');
 const clot=s.layers.find(l=>l.id==='artifact-clot');
 const red=await read('/shapes/'+clot.token);
 await showTissuePreview({layer:clot,features:red.features});
 button(footer(),'Subtract from tissue').click();
 await until(()=>footer().textContent.includes('Tissue saved.')&&!tissue().brushBusy);
 const cut=await read('/shapes/'+tissue().token),cutArea=area(cut.features);
 assert(!clot.preview&&cutArea<detectedArea&&tissue().saved_file===file,'Subtract saves the reviewed artifact preview and updates the same Tissue file');
 assert(!s.layers.some(l=>l.id==='clean-tissue-regions')&&s.lastEdited===tissue(),'Subtraction creates no Clean layer and targets Tissue for Undo');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
 await until(()=>!tissue().saveBusy&&!tissue().savePending&&area(tissue().features)===detectedArea);
 assert(area((await read('/shapes/'+tissue().token)).features)===detectedArea,'Ctrl+Z restores and saves the Tissue before subtraction');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'y',ctrlKey:true,bubbles:true}));
 await until(()=>!tissue().saveBusy&&!tissue().savePending&&area(tissue().features)===cutArea);
 assert(area((await read('/shapes/'+tissue().token)).features)===cutArea,'Ctrl+Y reapplies and saves subtraction');
 button(tools(),'Image quality').click();await until(()=>tools().querySelector('[data-artifact-training=debris]')&&tools().querySelector('[aria-label="Work within"]'));
 assert(tools().querySelector('[aria-label="Work within"]').value==='tissue'&&(await read('/studio/browser-check/annotation_sources?stage=debris')).selection.mode==='tissue','The next artifact tab selects the same saved working Tissue');
 const quality=s.layers.find(l=>l.id==='artifact-debris');await until(()=>quality?.revision);
 const f={type:'Feature',properties:{classification:{name:'Image_quality'}},geometry:{type:'Polygon',coordinates:[[[170,0],[390,0],[390,390],[170,390],[170,0]]]}};
 await showTissuePreview({layer:quality,features:[f]});button(footer(),'Subtract from tissue').click();
 await until(()=>footer().textContent.includes('Tissue saved.')&&!tissue().brushBusy&&area(tissue().features)<cutArea);
 assert(area((await read('/shapes/'+tissue().token)).features)<cutArea,'Image quality subtracts from the already-saved clot result');
 stage('Classification');await until(()=>tools().querySelector('[aria-label="Work within"]'));
 assert(tools().querySelector('[aria-label="Work within"]').value==='tissue','Pixel classes inherits the current Tissue choice');
 stage('Tissue');await until(()=>tools().querySelector('[aria-label="Tissue annotation"]'));
 const history=await read('/shapes/'+tissue().token+'/history');
 assert(history.entries.filter(e=>e.label==='Subtracted artifacts').length===2,'Both subtraction steps remain in durable History');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}},100);
</script>
"""
if __name__=='__main__':main(DRIVER,studio=True,timeout=240,setup=setup)
