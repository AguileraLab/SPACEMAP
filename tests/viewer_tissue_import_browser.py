"""A loaded GeoJSON class becomes editable original Tissue without changing its source."""
from viewer_threshold_browser import main
from viewer_artifact_review_browser import setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 const read=p=>fetch(api(p)).then(r=>r.json());
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.some(l=>l.id==='tissue-regions'&&l.revision));
 button(document.querySelector('.studio-stage-tabs'),'Tissue').click();
 const feature=(name,x0,y0,x1,y1)=>({type:'Feature',properties:{classification:{name}},geometry:{type:'Polygon',coordinates:[[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]}});
 await importAnnotationCopy({name:'QuPath',features:[feature('tissue',30,30,170,330),feature('tissue',230,30,350,330),feature('tumor',350,350,380,380)]});
 const imported=s.layers.find(l=>l.annotation_group==='imported');
 await until(()=>imported?.revision);
 const before=await read('/shapes/'+imported.token),original=(await read('/shapes/'+s.layers.find(l=>l.id==='tissue-regions').token)).revision;
 imported.open=true;buildPanel();button(document.querySelector('[data-layer-id="'+imported.id+'"]'),'Use as Tissue…').click();
 await until(()=>document.querySelector('[aria-label="Tissue import annotation"]')?.value==='"tissue"');
 const dialog=document.querySelector('#tissue-import-dialog');
 assert(document.querySelector('[aria-label="Tissue import annotation"]').options.length===4,'Import lists each named class plus an explicit All polygons choice');
 button(dialog,'Use as Tissue').click();
 await until(()=>!document.querySelector('#tissue-import-dialog')&&s.layers.find(l=>l.id==='tissue-regions')?.preview);
 const tissue=s.layers.find(l=>l.id==='tissue-regions');
 assert(tissue.features.length===2&&tissue.features.every(f=>f.properties.tissue_source_class==='tissue'),'Only the selected tissue class opens in the Tissue panel');
 assert((await read('/shapes/'+tissue.token)).revision===original,'Import remains a preview until Save polygons');
 await until(()=>tools().querySelector('[aria-label="Tissue annotation"]'));
 assert(tools().querySelector('[aria-label="Tissue annotation"]').value==='tissue','Import selects the working Tissue annotation');
 button(document.querySelector('.studio-runbar'),'Save polygons').click();await until(()=>!tissue.preview&&!tissue.saveBusy);
 assert((await read('/shapes/'+tissue.token)).features.length===2,'The footer saves imported Tissue as editable original polygons');
 assert((await read('/shapes/'+imported.token)).revision===before.revision,'The external annotation copy remains intact');
 button(tools(),'Import tissue…').click();await until(()=>document.querySelector('#tissue-import-dialog'));
 button(document.querySelector('#tissue-import-dialog'),'Cancel').click();
 assert(tissue.features.length===2,'Cancel preserves Tissue');
 button(document.querySelector('.studio-stage-tabs'),'Artifacts').click();
 await until(()=>tools().querySelector('[aria-label="Work within"]'));
 assert(tools().querySelector('[aria-label="Work within"]').value==='tissue','Artifact detection can use the imported Tissue directly');
 const manifest=await read('/studio/browser-check/annotation_workflow');
 assert(Object.values(manifest.assets).some(a=>a.sources?.some(x=>x.tissue_source_class==='tissue'&&x.tissue_source_revision===before.revision)),'The workflow record links imported Tissue to its exact source class and revision');
 const source=await read('/studio/browser-check/annotation_sources?source_token='+imported.token+'&image_token='+s.layers.find(l=>l.type==='image').token);
 const bad=await fetch(api('/studio/browser-check/annotation_sources'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'use_as_tissue',image_token:s.layers.find(l=>l.type==='image').token,source_token:imported.token,source_revision:'stale',class_name:'tissue'})});
 assert(bad.status===400&&source.revision===before.revision,'Import rejects an outdated source revision');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}},100);
</script>
"""
if __name__ == '__main__': main(DRIVER, studio=True, timeout=180, setup=setup)
