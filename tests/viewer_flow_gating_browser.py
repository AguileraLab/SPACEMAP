"""Real Studio Flow gating, plot interactions, Data/image sync and saved labels."""
from viewer_threshold_browser import main


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1600;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(el,v)=>{el.value=v;el.dispatchEvent(new Event('change'));};
const app=()=>document.querySelector('.flow-app');
const button=(text,within=app())=>[...within.querySelectorAll('button')].find(b=>b.textContent===text);
const endpoint='/spacemap-viewer/api/studio/browser-check/flow_gating';
const request=async body=>{const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error);return j;};
const info=async()=>await(await fetch(endpoint)).json();
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
  assert(location.protocol==='http:'&&!isSecureContext,'Full workflow runs on a non-secure HTTP origin');
  assert(typeof crypto.randomUUID==='undefined','Workflow does not rely on secure-context UUID support');
  await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
  window.open=()=>null; // exercise the popup-blocked path, then use the same docked workspace
  change(document.querySelector('[aria-label="Phenotyping method"]'),'flow');
  await until(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Dock below'));
  [...document.querySelectorAll('button')].find(b=>b.textContent==='Dock below').click();
  await until(()=>app()?.querySelector('.flow-chart .main-svg'));
  assert(document.querySelector('[aria-label="Phenotyping method"]').options[document.querySelector('[aria-label="Phenotyping method"]').options.length-1].value==='flow','Flow gating is last in the method list');
  assert(app().querySelectorAll('.flow-resize-handle').length===4,'All four plot corners resize');
  let found=await info();const x=found.numeric.find(n=>n==='DAPI')||found.numeric[0],y=found.numeric.find(n=>n!==x)||x;
  const spec={id:'testplot',parent:'root',x:{column:x,scale:'linear'},y:{column:y,scale:'linear'},kind:'scatter',bins:20,sample:100,frequency:'percent',style:{font:13,width:700,height:550}};
  const values=await request({action:'plot',revision:found.workspace.revision,plot:spec});
  const a=values.points[0].x.slice().sort((a,b)=>a-b);const lo=a[0],hi=a[Math.floor(a.length/2)];
  const gate={id:'gate_test',name:'Treg',colour:'#e15759',parent:'root',kind:'range',x:{column:x,scale:'linear'},bounds:[lo,hi]};
  await request({action:'save',revision:found.workspace.revision,changes:{gates:[gate]}});
  await until(()=>app().querySelector('[data-population="gate_test"]'));
  showDataTab();await dataChoose('combined');await until(()=>dataView.queryReady);
  app().querySelector('[data-population="gate_test"] .flow-name').click();
  await until(()=>window.__state.layers.some(l=>l.flowSelection));
  const layer=window.__state.layers.find(l=>l.flowSelection);
  assert(window.__state.layers.filter(l=>l.type==='cells').length===1,'Gate picks reuse the existing cell layer');
  await until(()=>dataView.queryReady&&dataView.flowSelection);
  const count=(await info()).stats.find(s=>s.id==='gate_test').n;
  assert(dataView.result.matched===count,'Data rows match the exact selected gate');
  assert(layer.flowMode==='filter','Gate click filters the image');
  assert(layer.flowPopulation.colour==='#e15759','Image legend uses the exact population colour');
  const fetchNow=window.fetch;let rootRequested=false;
  window.fetch=async(...args)=>{let body;try{body=JSON.parse(args[1]?.body||'{}');}catch(e){}
    if(body?.action==='select'&&body.populations?.[0]==='root'){rootRequested=true;const response=await fetchNow(...args);await delay(800);return response;}return fetchNow(...args);};
  app().querySelector('[data-population="root"] .flow-name').click();await until(()=>rootRequested);
  app().querySelector('[data-population="gate_test"] .flow-name').click();await delay(1300);window.fetch=fetchNow;
  assert(layer.flowPopulation.name==='Treg','A delayed population request cannot replace the newer selection');
  const row=app().querySelector('[data-population="gate_test"]');row.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:150,clientY:160}));
  assert(!!button('Edit / rename…',app().querySelector('.flow-menu')),'Right-click population exposes editing actions');
  button('Edit / rename…',app().querySelector('.flow-menu')).click();
  await until(()=>app().querySelector('dialog [aria-label="Population name"]'));
  app().querySelector('dialog [aria-label="Population name"]').value='Candidate Treg';button('Save',app().querySelector('dialog')).click();
  await until(()=>app().querySelector('[data-population="gate_test"] .flow-name')?.textContent.includes('Candidate Treg'));
  const card=app().querySelector('.flow-plot-card');card.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:600,clientY:200}));
  assert(!!button('Export figure…',app().querySelector('.flow-menu')),'Plot context menu offers export');app().querySelector('.flow-menu').remove();
  change(card.querySelector('[aria-label="Plot type"]'),'histogram');
  await until(()=>card.querySelector('.flow-chart').data?.[0]?.type==='scatter'&&card.querySelector('.flow-plot-foot').textContent.includes('Exact counts'));
  assert(card.querySelector('.flow-chart').data[0].x.length===80,'Histogram uses shared exact bins');
  button('Settings',card).click();await until(()=>app().querySelector('dialog [aria-label="Scatter sample limit"]'));
  app().querySelector('[aria-label="Scatter sample limit"]').value=500;
  app().querySelector('[aria-label="Text size"]').value=17;button('Apply',app().querySelector('dialog')).click();
  await until(()=>!app().querySelector('dialog')&&card.querySelector('.flow-chart').layout.font.size===17);
  assert(true,'Figure style and display sample limit are editable');
  const chart=card.querySelector('.flow-chart');
  await Plotly.relayout(chart,{shapes:[...(chart.layout.shapes||[]),{type:'rect',x0:lo,x1:hi,y0:0,y1:10}]});
  await until(()=>app().querySelector('dialog [aria-label="Population name"]'));
  app().querySelector('dialog [aria-label="Population name"]').value='Nested interval';button('Save',app().querySelector('dialog')).click();
  await until(async()=>(await info()).workspace.gates.some(g=>g.name==='Nested interval'));
  const nested=(await info()).workspace.gates.find(g=>g.name==='Nested interval');
  assert(nested.parent==='gate_test'&&nested.kind==='range','Drawn histogram interval creates a named child gate');
  const svg=await Plotly.toImage(chart,{format:'svg',width:700,height:550});
  const exported=await request({action:'export',format:'pdf',svg,name:'browser-figure',manifest:{plot:spec}});
  assert(exported.file.endsWith('.pdf')&&exported.download.startsWith('data:application/pdf'),'Real Plotly figure exports to PDF');
  const labels=[...app().querySelectorAll('details')].find(d=>d.querySelector('summary')?.textContent==='Phenotype labels');labels.open=true;
  labels.querySelector('[aria-label="Assign Candidate Treg"]').checked=true;
  labels.querySelector('[aria-label="Output phenotype column"]').value='phenotype_flow';
  button('Preview labels',labels).click();await until(()=>!button('Apply labels',labels).disabled);
  button('Apply labels',labels).click();await until(()=>labels.querySelector('.flow-label-plan').textContent.includes('saved in Parquet'));
  assert((await info()).workspace.applied.write===count,'Apply labels writes the exact gate to phenotype_flow');
  await until(()=>dataView.result?.columns.includes('phenotype_flow'));
  assert(dataView.result.kinds.phenotype_flow==='group','New output is immediately categorical in Data');
  card.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:600,clientY:200}));button('Freeze to layout',app().querySelector('.flow-menu')).click();
  await until(()=>app().querySelector('.flow-layout-panel .main-svg'));
  assert((await info()).workspace.layouts[0].panels[0].snapshot.histogram.series.length>0,'Publication page saves independent calculated snapshots');
  button('Style',app().querySelector('.flow-layout-panel')).click();await until(()=>app().querySelector('[aria-label="Panel text size"]'));
  app().querySelector('[aria-label="Panel text size"]').value=21;button('Apply',app().querySelector('dialog')).click();
  await until(()=>app().querySelector('.flow-layout-panel .flow-chart')?.layout?.font?.size===21);
  assert(card.querySelector('.flow-chart').layout.font.size===17,'Publication styling edits the snapshot independently of the live plot');
  button('Plots').click();assert(!!app().querySelector('.flow-plot-card'),'Plots can be restored from Layout');
  button('Transform',card).click();await until(()=>app().querySelector('dialog [aria-label="Transformation name"]'));
  const recipe=app().querySelector('dialog');recipe.querySelector('[aria-label="Transformation name"]').value='My ordered scale';
  change(recipe.querySelector('[aria-label="Step 1 operation"]'),'zscore');button('+ Add step',recipe).click();
  change(recipe.querySelector('[aria-label="Step 2 operation"]'),'cdf');button('+ Add step',recipe).click();
  change(recipe.querySelector('[aria-label="Step 3 operation"]'),'nlog');
  assert(recipe.querySelectorAll('[aria-label$="operation"]').length===3,'Transformation builder adds independently ordered steps');
  button('Save transformation',recipe).click();await until(()=>!app().querySelector('dialog'));
  chart.querySelector('.xtitle').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:400,clientY:500}));button('My ordered scale',app().querySelector('.flow-transform-menu')).click();
  await until(()=>card.querySelector('.flow-chart').layout.xaxis.title.text.includes('My ordered scale'));
  assert((await info()).workspace.plots[0].x.pipeline.map(s=>s.op).join(',')==='zscore,cdf,nlog','Saved plot retains the chosen Z1 → CDF → NLog order');
  card.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:600,clientY:200}));button('Duplicate plot',app().querySelector('.flow-menu')).click();
  await until(()=>app().querySelectorAll('.flow-plot-card .main-svg').length>=2);
  const second=app().querySelectorAll('.flow-plot-card')[1];
  change(second.querySelector('[aria-label="Plot type"]'),'ridgeline');await until(()=>second.querySelector('.flow-chart').layout?.yaxis?.title?.text?.includes('Offset'));
  assert(card.querySelector('[aria-label="Plot type"]').value==='histogram','Several plots keep independent presentation and transformation settings');
  const checked=app().querySelector('[aria-label="Live view"]');checked.checked=true;checked.dispatchEvent(new Event('change'));
  document.dispatchEvent(new CustomEvent('spacemap-cells-picked',{detail:{layer,labels:[]}}));
  await until(()=>app().querySelector('.flow-plot-foot').textContent.includes('0 in live scope'));
  assert(true,'An empty live cell selection produces zero cells');
  const detached=document.createElement('iframe');detached.style='position:fixed;width:1100px;height:800px;left:0;top:0;z-index:100000';
  detached.src='/spacemap-viewer/api/window/flow-gating.html?session=browser-check&api=/spacemap-viewer/api';document.body.append(detached);
  await until(()=>detached.contentDocument?.querySelector('[data-population="gate_test"]'));
  await until(()=>detached.contentDocument?.querySelector('.flow-plot-foot')?.textContent.includes('0 in live scope'));
  assert(true,'Detached workspace page restores gates and receives live viewer selections');detached.remove();
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(e+'\n'+e.stack+'\n'+app()?.querySelector('.flow-status')?.textContent));}},100);
</script>
"""

if __name__ == '__main__':
    main(driver=DRIVER,studio=True,timeout=240,screenshot=False,browser_host='flow-gating.test')
