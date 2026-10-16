"""Classification controls, linked heatmap review and both preserved Artifact scopes."""
from viewer_threshold_browser import main
from viewer_artifact_review_browser import setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<2000;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools'),footer=()=>document.querySelector('.studio-runbar');
 const read=p=>fetch(api(p)).then(r=>r.json()),stage=name=>button(document.querySelector('.studio-stage-tabs'),name).click();
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.find(l=>l.type==='image')?.name==='markers.ome.tiff'&&s.layers.find(l=>l.id==='tissue-regions')?.revision);
 stage('Artifacts');await until(()=>tools().querySelector('[aria-label="Work within"]:not(:disabled)'));
 assert(tools().querySelector('[aria-label="Work within"]').value==='tissue','Blood clots keeps Work within with Tissue selected');
 button(tools(),'Image quality').click();await until(()=>tools().querySelector('[data-artifact-training=debris]')&&tools().querySelector('[aria-label="Work within"]:not(:disabled)'));
 assert(tools().querySelector('[aria-label="Work within"]').options.length>1,'Image quality keeps selectable Work within alternatives');
 const originalFetch=window.fetch;
 window.fetch=async(url,options)=>String(url).endsWith('/tissue_versions')?new Response(JSON.stringify({error:'Simulated old server'}),{status:400,headers:{'Content-Type':'application/json'}}):originalFetch(url,options);
 button(tools(),'Blood clots').click();await until(()=>tools().textContent.includes('Simulated old server'));
 assert(!!tools().querySelector('[aria-label="Work within"]:not(:disabled)'),'A failed working-layer request never removes Work within');
 window.fetch=originalFetch;
 stage('Classification');
 const barRow=label=>Array.from(footer().querySelectorAll('.row')).find(r=>r.querySelector('.lbl')?.textContent===label);
 const classifyNow=()=>button(barRow('Classify:'),'This image');
 const ready=()=>tools().querySelectorAll('.classification-row').length>0&&!classifyNow().disabled;
 await until(()=>/Not classified in this area yet/.test(tools().textContent));
 assert(!tools().querySelector('.classification-row')&&button(barRow('Classify:'),'Selected'),'Opening Classification does not classify by itself; Classify: This image | Selected');
 classifyNow().click();
 await until(ready);
 assert(tools().querySelector('[aria-label="Work within"]').value==='tissue'&&tools().querySelector('[aria-label="Classification scale"]'),'Classification has the same Work within and Scale controls');
 assert(!tools().querySelector('[aria-label="Selected markers only"]').checked,'All image channels are the default');
 const initialCount=tools().querySelectorAll('.classification-row').length;
 assert(initialCount>1&&document.querySelector('svg[aria-label=Classification]'),'Classification draws the list and interactive heatmap');
 const field=(label,value)=>{const el=tools().querySelector('[aria-label="'+label+'"]');el.value=value;el.dispatchEvent(new Event('change'));};
 field('Top markers to show','8');
 assert(tools().querySelector('[aria-label="Top markers to show"]').value==='8','The number of top markers is adjustable');
 const id=tools().querySelector('.classification-row').dataset.class;
 const name=tools().querySelector('[aria-label="Name of class '+id+'"]');name.value='Reviewed tumor';
 tools().querySelector('[aria-label="Accept name of class '+id+'"]').click();await until(ready);
 assert(document.querySelector('svg[aria-label=Classification] [data-row="'+id+'"] .label').textContent.includes('Reviewed'),'Accepting a name updates the heatmap');
 const summary=()=>document.querySelector('#studio-figures [aria-label="Heatmap summary"]');
 assert(summary()?.value==='median','Annotation Classification defaults to Median');
 summary().value='mean';summary().dispatchEvent(new Event('change'));
 await until(()=>ready()&&summary()?.value==='mean'&&/Mean · Arcsinh/.test(document.querySelector('#studio-figures').textContent));
 assert(tools().querySelector('[aria-label="Name of class '+id+'"]').value==='Reviewed tumor'&&tools().querySelectorAll('.classification-row').length===initialCount,'Mean updates the heatmap while keeping class names and membership');
 assert(document.querySelector('svg[aria-label=Classification] rect title').textContent.includes('Mean scaled pixel value (before Z-score)'),'Classification hover labels identify the actual scaled pixel values');
 summary().value='median';summary().dispatchEvent(new Event('change'));
 await until(()=>ready()&&summary()?.value==='median'&&/Median · Arcsinh/.test(document.querySelector('#studio-figures').textContent));
 const lines=()=>Array.from(document.querySelectorAll('svg[aria-label=Classification] g.row'));
 const ids=lines().slice(0,2).map(n=>n.dataset.row);
 lines()[0].dispatchEvent(new MouseEvent('click',{bubbles:true}));await until(ready);
 document.querySelector('svg[aria-label=Classification] [data-row="'+ids[1]+'"]').dispatchEvent(new MouseEvent('click',{bubbles:true,ctrlKey:true}));await until(ready);
 assert(!button(tools(),'Merge selected').disabled,'Ctrl-clicking multiple heatmap rows enables Merge');
 button(tools(),'Merge selected').click();await until(()=>ready()&&tools().querySelectorAll('.classification-row').length===initialCount-1);
 const merged=tools().querySelector('[aria-label^="Unmerge class"]');
 assert(!!merged&&lines().length===initialCount-1,'Merge changes the list, map and heatmap together');
 merged.click();await until(()=>ready()&&tools().querySelectorAll('.classification-row').length===initialCount);
 assert(tools().querySelector('[aria-label="Name of class '+id+'"]').value==='Reviewed tumor','Unmerge restores original clusters and their names');
 document.querySelector('[aria-label="Cluster heatmap X axis"]').click();
 document.querySelector('[aria-label="Cluster heatmap Y axis"]').click();
 const palette=document.querySelector('[aria-label="Heatmap colours"]');const old=document.querySelector('svg[aria-label=Classification] g.row rect[fill]')?.getAttribute('fill');
 palette.value='viridis';palette.dispatchEvent(new Event('change'));
 assert(document.querySelector('[aria-label="Heatmap colours"]').value==='viridis'&&document.querySelector('[aria-label="Cluster heatmap X axis"]').checked&&document.querySelector('[aria-label="Cluster heatmap Y axis"]').checked,'Heatmap colour and independent axis clustering controls work');
 const marker=s.layers.find(l=>l.type==='image');marker.channels.forEach(c=>c.visible=c.name==='CD3');
 tools().querySelector('[aria-label="Selected markers only"]').click();
 await until(()=>/Settings changed: press Classify/.test(tools().textContent));
 assert(document.querySelectorAll('svg[aria-label=Classification] text.column').length>1,'Changing a setting does not classify again by itself');
 classifyNow().click();await until(()=>ready()&&document.querySelectorAll('svg[aria-label=Classification] text.column').length===1);
 assert(document.querySelectorAll('svg[aria-label=Classification] text.column').length===1,'Selected markers only uses the checked image marker');
 field('Classification scale','2');classifyNow().click();await delay(100);await until(ready);
 const savedSettings=await read('/studio/browser-check/classification');
 assert(savedSettings.options.scale===2&&savedSettings.options.markers.join(',')==='CD3','Scale and selected marker choices persist in project JSON');
 let aiPayload=null;
 window.fetch=async(url,options)=>{
  if(String(url).endsWith('/classification')&&options?.body){const body=JSON.parse(options.body);
   if(body.action==='suggest'){aiPayload=body;const response=await originalFetch(url,{...options,body:JSON.stringify({...body,action:'run'})});const data=await response.json();
    data.suggestions={say:'Review these marker patterns.',images_used:2,names:[{id:body.selected[0],name:'AI review candidate',reason:'Enrichment compared with the other clusters.'}],merges:[]};
    return new Response(JSON.stringify(data),{status:200,headers:{'Content-Type':'application/json'}});
   }
  }return originalFetch(url,options);
 };
 const aiRow=tools().querySelector('.classification-row'),aiId=aiRow.dataset.class;
 button(aiRow,'Ask AI').click();await until(()=>ready()&&tools().querySelector('[aria-label="Name of class '+aiId+'"]').value==='AI review candidate');
 assert(aiPayload.images.length===2&&aiPayload.selected.includes(Number(aiId)),'Ask AI attaches the viewer, exact class map and selected cluster');
 assert(!document.querySelector('svg[aria-label=Classification] [data-row="'+aiId+'"] .label').textContent.includes('AI review'),'AI names stay suggestions until accepted');
 tools().querySelector('[aria-label="Accept name of class '+aiId+'"]').click();await until(ready);
 assert(document.querySelector('svg[aria-label=Classification] [data-row="'+aiId+'"] .label').textContent.includes('AI review'),'Accepting an AI proposal updates the heatmap');
 window.fetch=originalFetch;
 // Showing one class is drawn here, without asking the server.
 let asked=0;window.fetch=async(url,options)=>{if(String(url).endsWith('/classification'))asked++;return originalFetch(url,options);};
 lines()[0].dispatchEvent(new MouseEvent('click',{bubbles:true}));await delay(300);
 assert(asked===0&&s.rasters?.classes,'Picking a class in the heatmap recolours the map at once, with no server request');
 window.fetch=originalFetch;
 // Leaving and coming back shows the saved classes without classifying again.
 stage('Tissue');await delay(300);stage('Classification');
 await until(()=>ready()&&/saved run/.test(tools().textContent));
 assert(tools().querySelector('[aria-label="Name of class '+aiId+'"]').value==='AI review candidate','The saved classes come back with their names, without classifying again');
 tools().querySelector('[aria-label="Smallest island"]').value='200';
 button(barRow('Save polygons:'),'This image').click();await until(()=>s.layers.find(l=>l.id==='pixel-classes')?.features.length&&ready());
 assert(/islands below 200/.test(footer().textContent),'Save polygons takes Smallest hole and Smallest island');
 const layer=s.layers.find(l=>l.id==='pixel-classes');
 assert(layer.editable&&layer.visible&&layer.features.length>0,'Save polygons creates visible editable Classification annotations');
 button(document.querySelector('.studio-stage-tabs'),'Zones').click();
 await until(()=>button(tools(),'Add a pool')||/Export the pixel classes first/.test(tools().textContent));
 assert(button(tools(),'Add a pool')&&!/Export the pixel classes first/.test(tools().textContent),'Zones starts from the saved classes (their domains)');
 const stored=await read('/shapes/'+layer.token);
 assert(stored.features.every(f=>f.properties.classification_run),'Saved polygons retain the exact classification run');
 // Selected: the ticked images, one after another in the background.
 stage('Classification');await until(ready);
 field('Number of classes','4');
 const tick=tools().querySelector('tr[data-image="markers.ome.tiff"] input[type=checkbox]');if(!tick.checked)tick.click();
 button(barRow('Classify:'),'Selected').click();
 let job=null;
 for(let i=0;i<900;i++){const jobs=(await read('/studio/browser-check/jobs')).jobs||[];job=jobs.find(j=>j.kind==='classify_images');if(job&&/done|failed/.test(job.status))break;await delay(200);}
 assert(job&&job.status==='done'&&job.result.count===1&&job.result.images[0].image==='markers.ome.tiff'&&job.result.images[0].classes<=4,'Selected classifies the ticked images as a job: '+JSON.stringify(job&&job.result));
 stage('Tissue');await delay(300);stage('Classification');
 await until(()=>ready()&&/saved run/.test(tools().textContent)&&tools().querySelectorAll('.classification-row').length<=4);
 assert(true,'A finished image opens with the classes the job found');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}},100);
</script>
"""
if __name__=='__main__':main(DRIVER,studio=True,timeout=300,setup=setup)
