"""TLS progress must update status without rebuilding the image/table workspace."""
from viewer_threshold_browser import main
from viewer_studio_tls_browser import setup


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<900;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const check=(ok,message)=>{if(!ok)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const button=(parent,text)=>[...parent.querySelectorAll('button')].find(b=>b.textContent===text);
const realFetch=window.fetch;let job=null, other=null, polls=0, inputsCalls=0, overviewCalls=0, tlsCalls=0;
window.fetch=async(resource,options)=>{
 const path=typeof resource==='string'?resource:resource.url;
 if(path.endsWith('/jobs')&&(!options?.method||options.method==='GET')){
  polls++;return new Response(JSON.stringify({jobs:[job?{...job,message:`${job.image} · region 1: Measuring boundary distances: ${polls}/100`,progress:polls/100}:null,other].filter(Boolean)}),{headers:{'Content-Type':'application/json'}});
 }
 if(path.endsWith('/spatial/inputs'))inputsCalls++;
 if(path.includes('/spatial/overview'))overviewCalls++;
 if(path.endsWith('/spatial/tls_result'))tlsCalls++;
 return realFetch(resource,options);
};
setTimeout(async()=>{try{
 const state=window.__state,tools=()=>document.querySelector('#studio-tools');
 await until(()=>state.imagesReady&&state.cellsReady&&document.querySelector('[data-step=tab8]'));
 document.querySelector('[data-step=tab8]').click();
 await until(()=>button(tools(),'TLS candidates'));button(tools(),'TLS candidates').click();
 await until(()=>tools().querySelector('[aria-label="TLS maximum gap"]')&&tools().querySelector('[data-cell-count]')?.textContent==='40');
 document.querySelector('#split-marker-toggle').click();
 await until(()=>splitMarkers.cards.length&&splitMarkers.cards[0].canvas.width>0);
 const image=state.layers.find(l=>l.type==='image'),viewer=state.viewer, item=image.item;
 splitMarkers.point={x:100,y:100};splitMarkers.pinned=true;scheduleSplitMarkers();await delay(600);
 const card=splitMarkers.cards[0], canvas=card.canvas, pixels=canvas.toDataURL();
 const row=tools().querySelector('tr[data-roi="roi"]'), tick=row.querySelector('input');tick.click();
 const folder=tools().querySelector('[aria-label="Spatial images folder"]');
 folder.focus();folder.setSelectionRange(2,6);
 const initial={generation:state.initGeneration,inputsCalls,overviewCalls,tlsCalls};
 job={id:'progress-fixture',kind:'tls_candidates',status:'running',started:new Date().toISOString(),image:'roi',result:{},settings:{}};
 const before=polls;await until(()=>polls>=before+3);await delay(800);
 fetch('/result?progress='+encodeURIComponent(JSON.stringify({inputsReloaded:inputsCalls-initial.inputsCalls,summaryReloaded:overviewCalls-initial.overviewCalls,viewerReopened:state.initGeneration-initial.generation,rowReplaced:row!==tools().querySelector('tr[data-roi="roi"]'),markerReplaced:canvas!==splitMarkers.cards[0]?.canvas})));
 check(row===tools().querySelector('tr[data-roi="roi"]'),'Progress preserves the image row and its checked state');
 const working=row=>!row.querySelector('[data-working]').hidden;
 check(tick.checked&&working(row),'TLS region progress marks the correct image');
 check(document.activeElement===folder&&folder.selectionStart===2&&folder.selectionEnd===6,'Progress preserves source-field focus and selection');
 check(inputsCalls===initial.inputsCalls&&overviewCalls===initial.overviewCalls&&tlsCalls===initial.tlsCalls,'Progress does not reload inputs, summaries or TLS geometry');
 check(state.viewer===viewer&&image.item===item&&state.initGeneration===initial.generation,'Progress does not reopen the image or change the viewport');
 check(splitMarkers.cards[0].canvas===canvas&&canvas.toDataURL()===pixels&&splitMarkers.pinned&&splitMarkers.point.x===100,'Progress preserves rendered marker pixels and the pinned location');
 job.image='roi2';const next=polls;await until(()=>polls>next);await delay(200);
 check(!working(row)&&working(tools().querySelector('tr[data-roi="roi2"]')),'Working status follows the next image without navigation');
 other={...job,id:'concurrent-fixture',kind:'neighbourhoods',message:'roi: Calculating neighbours'};
 const concurrent=polls;await until(()=>polls>concurrent);await delay(100);
 job.status='failed';job.finished=new Date().toISOString();const end=polls;await until(()=>polls>end&&overviewCalls>initial.overviewCalls);await delay(1000);
 check(working(row)&&!working(tools().querySelector('tr[data-roi="roi2"]')),'Finishing refreshes results while another job continues');
 check(row===tools().querySelector('tr[data-roi="roi"]')&&folder===tools().querySelector('[aria-label="Spatial images folder"]'),'Completion refresh preserves image rows and directory controls');
 check(row.querySelector('[data-cell-count]').textContent==='40'&&inputsCalls===initial.inputsCalls,'Completion retains counts and does not rediscover source files');
 check(state.viewer===viewer&&canvas===splitMarkers.cards[0].canvas&&canvas.toDataURL()===pixels,'Completion leaves the image and marker previews intact');
 other.status='failed';other.finished=new Date().toISOString();const last=polls;await until(()=>polls>last);await delay(300);
 check(!working(row)&&!working(tools().querySelector('tr[data-roi="roi2"]')),'Final completion clears both working badges');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(error.stack||String(error)));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=90, setup=setup, screenshot=False)
