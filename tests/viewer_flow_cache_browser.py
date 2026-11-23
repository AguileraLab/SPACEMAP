"""New plots retain axes; Jobs clears generated files without removing plots."""
from viewer_threshold_browser import main

DRIVER=r'''<script>
const pause=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<1000;i++){if(await f())return;await pause(40);}throw Error('Timed out '+f);};
const check=(v,m)=>{if(!v)throw Error(m);fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
const app=()=>document.querySelector('.flow-app'),btn=(label,root=app())=>[...root.querySelectorAll('button')].find(e=>e.textContent===label);
(async()=>{try{
 await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 window.open=()=>null;change(document.querySelector('[aria-label="Phenotyping method"]'),'flow');btn('Dock below',document).click();
 await until(()=>app()?.querySelector('.flow-plot-foot')?.textContent.includes('Exact counts'));
 const first=app().querySelector('.flow-plot-card');
 change(first.querySelector('[aria-label="X measurement"]'),'M1');change(first.querySelector('[aria-label="Y measurement"]'),'M0');
 await until(()=>first.querySelector('.flow-plot-foot').textContent.includes('Exact counts')&&first.querySelector('.flow-chart').layout.xaxis.title.text.includes('M1'));
 btn('New plot',app().querySelector('.flow-toolbar')).click();
 await until(()=>app().querySelectorAll('.flow-plot-card').length===2);
 const second=app().querySelectorAll('.flow-plot-card')[1];
 check(second.querySelector('[aria-label="X measurement"]').value==='M1'&&second.querySelector('[aria-label="Y measurement"]').value==='M0','New plot retains the active marker axes');
 await until(()=>second.querySelector('.flow-plot-foot').textContent.includes('Exact counts'));
 check(JSON.stringify(first.querySelector('.flow-chart').data[0].x)===JSON.stringify(second.querySelector('.flow-chart').data[0].x),'Repeated plot uses the same exact analysis and display sample');
 const endpoint='/spacemap-viewer/api/studio/browser-check/flow_cache';
 const before=await(await fetch(endpoint)).json();check(before.files>0,'Plot results are cached on disk');
 document.querySelector('[data-tab="jobs"]').click();
 const caches=document.querySelector('[data-performance-caches]');caches.open=true;
 await until(()=>caches.querySelector('[data-cache-group=flow_plots]'));
 caches.querySelector('[data-cache-group=flow_plots]').open=true;
 let asked='';window.confirm=message=>{asked=message;return false;};
 btn('Clear Flow cache',document.querySelector('#studio-jobs')).click();
 check(asked.includes('Gates, labels and figures are kept')&&(await(await fetch(endpoint)).json()).files===before.files,'Cancelling cache cleanup keeps cached results');
 window.confirm=()=>true;btn('Clear Flow cache',document.querySelector('#studio-jobs')).click();
 await until(()=>document.querySelector('#studio-jobs').textContent.includes('Cleared · rebuilds as needed.'));
 check((await(await fetch(endpoint)).json()).files===0,'Jobs clears the generated cache');
 const state=await(await fetch('/spacemap-viewer/api/studio/browser-check/flow_gating?metadata=1')).json();
 check(state.workspace.plots.length===2&&app().querySelectorAll('.flow-plot-card').length===2,'Cache cleanup preserves saved and displayed plots');
 btn('New plot',app().querySelector('.flow-toolbar')).click();
 await until(()=>app().querySelectorAll('.flow-plot-card')[2]?.querySelector('.flow-plot-foot').textContent.includes('Exact counts'));
 check((await(await fetch(endpoint)).json()).files>0,'Plots rebuild the cache after clearing');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks:7})));
}catch(e){fetch('/result?error='+encodeURIComponent(e.stack||String(e)));}})();
</script>'''

if __name__=='__main__':
    main(driver=DRIVER,studio=True,screenshot=False,timeout=150,browser_host='flow-gating.test')
