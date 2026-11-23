"""Numbered transformation recipes and independent axis context-menu application."""
from viewer_threshold_browser import main

DRIVER=r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<800;i++){if(await f())return;await delay(50);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change',{bubbles:true}));},type=(e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));};
const app=()=>document.querySelector('.flow-app'),button=(text,root=app())=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
const endpoint='/spacemap-viewer/api/studio/browser-check/flow_gating',info=async()=>await(await fetch(endpoint)).json();
const post=async body=>{const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),v=await r.json();if(!r.ok)throw Error(v.error);return v;};
const axisMenu=(chart,axis)=>{const title=chart.querySelector('.'+axis+'title'),r=title.getBoundingClientRect(),hit=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2);check(hit===title||title.contains(hit),'Axis title receives the right-click');hit.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:r.left+r.width/2,clientY:r.top+r.height/2}));return app().querySelector('.flow-menu');};
const close=(a,b)=>a.length===b.length&&a.every((x,i)=>Math.abs(x-b[i])<1e-8);
(async()=>{try{
 await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 await post({action:'save',revision:(await info()).workspace.revision,changes:{recipes:[{id:'legacy',name:'Legacy Z2',pipeline:[{op:'row_zscore'}],markers:['M0','M1']}]}});
 window.open=()=>null;change(document.querySelector('[aria-label="Phenotyping method"]'),'flow');button('Dock below',document).click();await until(()=>app()?.querySelector('.flow-chart .main-svg'));
 const card=app().querySelector('.flow-plot-card'),chart=card.querySelector('.flow-chart');change(card.querySelector('[aria-label="X measurement"]'),'M0');change(card.querySelector('[aria-label="Y measurement"]'),'M1');
 await until(()=>chart.layout.xaxis.title.text.includes('M0')&&chart.layout.yaxis.title.text.includes('M1'));button('Expand',card).click();await delay(250);
 const rawX=Array.from(chart.data[0].x),rawY=Array.from(chart.data[0].y);let plotCalls=0;const fetchNow=window.fetch;window.fetch=(...args)=>{try{if(JSON.parse(args[1]?.body||'{}').action==='plot')plotCalls++;}catch(e){}return fetchNow(...args);};
 button('Transform',card).click();let d=app().querySelector('dialog');
 check(!d.querySelector('[aria-label="Normalization reference"]')&&!d.querySelector('[aria-label="Recipe target"]')&&!button('Apply to plot',d),'Builder has no reference or axis selection');
 check(d.querySelectorAll('.flow-transform-step').length===1&&d.querySelector('.flow-step-number').textContent==='1','Builder starts with a clearly numbered step');
 check(!d.querySelector('[aria-label="Add Z2 marker"]')&&![...d.querySelectorAll('details')].some(e=>e.open),'Unneeded marker controls are absent and step settings start collapsed');
 type(d.querySelector('[aria-label="Transformation name"]'),'Arc Z');button('+ Add step',d).click();change(d.querySelector('[aria-label="Step 2 operation"]'),'zscore');button('+ Add step',d).click();change(d.querySelector('[aria-label="Step 3 operation"]'),'cdf');
 d.querySelector('[aria-label="Move step 3 up"]').click();check([...d.querySelectorAll('[aria-label$="operation"]')].map(e=>e.value).join()==='arcsinh,cdf,zscore','Steps can be reordered directly');
 d.querySelector('[aria-label="Move step 2 down"]').click();d.querySelector('[aria-label="Remove step 3"]').click();
 check([...d.querySelectorAll('.flow-step-number')].map(e=>e.textContent).join()==='1,2','Removing a step renumbers the remaining list');
 d.querySelector('[data-step="1"] details').open=true;type(d.querySelector('[aria-label="Step 1 label"]'),'First scale');type(d.querySelector('[aria-label="Step 1 Value"]'),'0');button('Save transformation',d).click();await until(()=>d.querySelector('[role="status"]').textContent.includes('Cofactor'));
 check((await info()).workspace.recipes.length===1,'Invalid cofactor is rejected before saving');type(d.querySelector('[aria-label="Step 1 Value"]'),'5');button('Save transformation',d).click();await until(()=>!app().querySelector('dialog'));
 let state=await info(),saved=state.workspace.recipes.find(r=>r.name==='Arc Z');check(saved.pipeline.map(s=>s.op).join()==='arcsinh,zscore'&&saved.pipeline[0].name==='First scale','Named transformations retain ordered steps and optional labels');
 check(plotCalls===0&&close(Array.from(chart.data[0].x),rawX),'Saving a transformation neither runs analysis nor changes an axis');
 let menu=axisMenu(chart,'x');check(!!button('Arc Z',menu)&&!!button('Legacy Z2',menu)&&!!button('Stored values',menu),'Axis menu lists all saved transformations and stored values');button('Arc Z',menu).click();
 await until(()=>chart.layout.xaxis.title.text.includes('Arc Z'));await until(async()=>(await info()).workspace.plots[0].x.recipe_id===saved.id);
 const expected=rawX.map(v=>Math.asinh(v/5)),mean=expected.reduce((a,b)=>a+b,0)/expected.length,sd=Math.sqrt(expected.reduce((a,b)=>a+(b-mean)**2,0)/expected.length);
 check(close(Array.from(chart.data[0].x),expected.map(v=>(v-mean)/sd)),'Axis menu applies the requested sequential math');
 state=await info();check(state.workspace.plots[0].x.reference_population==='root'&&state.workspace.plots[0].x.reference_n===40,'Normalization automatically fits Analysis cells');
 check(close(Array.from(chart.data[0].y),rawY),'Applying to X preserves Y');
 menu=axisMenu(chart,'y');button('New transformation…',menu).click();await until(()=>app().querySelector('dialog'));d=app().querySelector('dialog');type(d.querySelector('[aria-label="Transformation name"]'),'Row scale');change(d.querySelector('[aria-label="Step 1 operation"]'),'row_zscore');
 check(d.querySelector('[data-step="1"] details [aria-label="Step 1 markers"]').value==='axes','Z2 marker configuration lives inside its step and defaults to plot axes');
 button('Save transformation',d).click();await until(()=>!app().querySelector('dialog'));menu=axisMenu(chart,'y');button('Row scale',menu).click();await until(()=>chart.layout.yaxis.title.text.includes('Row scale'));
 check(close(Array.from(chart.data[0].y),rawY.map((v,i)=>v===rawX[i]?0:v>rawX[i]?1:-1)),'Z2 on Y uses the current plot markers without another selection');
 check(close(Array.from(chart.data[0].x),expected.map(v=>(v-mean)/sd)),'X and Y keep independent named transformations');
 change(card.querySelector('[aria-label="Y measurement"]'),'x_centroid');await until(()=>chart.layout.yaxis.title.text.includes('x_centroid'));
 const centroid=rawX.map(v=>40*(((v-1)/10)%8)+20);
 check(close(Array.from(chart.data[0].y),centroid.map((v,i)=>v===rawX[i]?0:v>rawX[i]?1:-1)),'Changing a Z2 axis marker updates its plot-axis marker set correctly');
 change(card.querySelector('[aria-label="Y measurement"]'),'M1');await until(()=>chart.layout.yaxis.title.text.includes('M1'));
 check(close(Array.from(chart.data[0].y),rawY.map((v,i)=>v===rawX[i]?0:v>rawX[i]?1:-1)),'Changing the marker back restores the expected Z2 values');
 menu=axisMenu(chart,'x');button('Stored values',menu).click();await until(()=>chart.layout.xaxis.title.text.includes('linear'));
 check(close(Array.from(chart.data[0].x),rawX)&&chart.layout.yaxis.title.text.includes('Row scale'),'Stored values removes only the selected axis transformation');
 menu=axisMenu(chart,'x');button('Legacy Z2',menu).click();await until(()=>chart.layout.xaxis.title.text.includes('Legacy Z2'));
 check(close(Array.from(chart.data[0].x),rawX.map((v,i)=>v===rawY[i]?0:v>rawY[i]?1:-1)),'Previously saved Z2 recipes remain usable');
 button('Transform',card).click();d=app().querySelector('dialog');change(d.querySelector('[aria-label="Saved transformations"]'),saved.id);
 d.querySelector('[data-step="1"] details').open=true;type(d.querySelector('[aria-label="Step 1 Value"]'),'10');button('Save transformation',d).click();await until(()=>!app().querySelector('dialog'));
 state=await info();check(state.workspace.recipes.find(r=>r.id===saved.id).version===2&&chart.layout.xaxis.title.text.includes('Legacy Z2'),'Editing a named transformation preserves its identity and does not change plots');
 menu=axisMenu(chart,'x');button('Arc Z',menu).click();await until(async()=>(await info()).workspace.plots[0].x.recipe_version===2);
 const revised=rawX.map(v=>Math.asinh(v/10)),m=revised.reduce((a,b)=>a+b,0)/revised.length,s=Math.sqrt(revised.reduce((a,b)=>a+(b-m)**2,0)/revised.length);
 check(close(Array.from(chart.data[0].x),revised.map(v=>(v-m)/s)),'Reapplying uses the edited configuration');
 // The Z2 marker scopes belong to their individual steps, not a global chooser.
 menu=axisMenu(chart,'x');button('New transformation…',menu).click();await until(()=>app().querySelector('dialog'));d=app().querySelector('dialog');type(d.querySelector('[aria-label="Transformation name"]'),'Two row steps');change(d.querySelector('[aria-label="Step 1 operation"]'),'row_zscore');
 change(d.querySelector('[aria-label="Step 1 markers"]'),'custom');button('+ Add step',d).click();change(d.querySelector('[aria-label="Step 2 operation"]'),'row_zscore');change(d.querySelector('[aria-label="Step 2 markers"]'),'custom');
 d.querySelector('[aria-label="Step 2 marker M1"]').click();d.querySelector('[aria-label="Step 2 marker x_centroid"]').click();button('Save transformation',d).click();await until(()=>!app().querySelector('dialog'));
 const two=(await info()).workspace.recipes.find(r=>r.name==='Two row steps');check(two.pipeline[0].markers.join()==='M0,M1'&&two.pipeline[1].markers.join()==='M0,x_centroid','Each Z2 step saves its own marker set');
 change(card.querySelector('[aria-label="Plot type"]'),'histogram');await until(()=>chart.layout.yaxis.title.text==='Cells / bin (%)');
 menu=axisMenu(chart,'y');check(!!button('Count',menu)&&!button('Arc Z',menu),'Histogram Y offers frequency controls instead of silently transforming a hidden marker');button('Count',menu).click();await until(()=>chart.layout.yaxis.title.text==='Cells / bin');
 menu=axisMenu(chart,'x');check(!!button('Arc Z',menu)&&!!button('Two row steps',menu),'Histogram X exposes the same named transformation library');app().querySelector('.flow-menu').remove();
 const frame=document.createElement('iframe');frame.style='position:fixed;inset:0;width:1200px;height:900px;z-index:100000';frame.src='/spacemap-viewer/api/window/flow-gating.html?session=browser-check&api=/spacemap-viewer/api';document.body.append(frame);
 await until(()=>frame.contentDocument?.querySelector('.flow-chart .xtitle'));const title=frame.contentDocument.querySelector('.xtitle');title.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:400,clientY:500}));
 check([...frame.contentDocument.querySelectorAll('.flow-transform-menu button')].some(b=>b.textContent==='Two row steps'),'Saved transformations return in axis menus after reopening');frame.remove();
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+app()?.querySelector('.flow-status')?.textContent));}})();
</script>'''

if __name__=='__main__':
    main(driver=DRIVER,studio=True,screenshot=False,timeout=240,browser_host='flow-gating.test')
