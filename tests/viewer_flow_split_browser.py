"""Population splits and direct colour edits through the real Studio browser UI."""
from viewer_threshold_browser import main


def setup(root):
    import numpy as np
    import pandas as pd
    from spacemap import dataio
    table=root/'cells.csv'
    frame=pd.read_csv(table)
    frame['timepoint']=['pre','post']*20
    frame['tumor']=np.arange(len(frame))%2
    frame.to_csv(table,index=False)
    dataio.ensure_fast_copy(table,force=True)


DRIVER=r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<800;i++){if(await f())return;await delay(50);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
const app=()=>document.querySelector('.flow-app'),button=(text,root=app())=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
const info=async()=>await(await fetch('/spacemap-viewer/api/studio/browser-check/flow_gating')).json();
let splitResponse=null,splitSerial=0;const actualFetch=window.fetch;window.fetch=async(...args)=>{const response=await actualFetch(...args);let body;try{body=JSON.parse(args[1]?.body||'{}');}catch(e){}if(body?.action==='split'){splitResponse=await response.clone().json();splitSerial++;}return response;};
const row=id=>app().querySelector('[data-population="'+id+'"]');
const confirmSplit=async()=>{const serial=splitSerial;button('Split',app().querySelector('dialog[open]')).click();await until(()=>splitSerial>serial&&!app().querySelector('dialog[open]'));};
const splitBy=async(id,column,confirmCategories=true)=>{
 const serial=splitSerial;
 row(id).dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:120,clientY:180}));
 button('Split by…',app().querySelector('.flow-menu')).click();
 await until(()=>app().querySelector('[aria-label="Split by column"] input'));
 const search=app().querySelector('[aria-label="Split by column"] input');search.value=column;search.dispatchEvent(new Event('input'));
 button(column,app().querySelector('.flow-split-menu')).click();
 await until(()=>splitSerial>serial);
 if(splitResponse.kind==='categorical'){await until(()=>app().querySelector('.flow-split-categories'));if(confirmCategories)await confirmSplit();}
};
(async()=>{try{
 await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 window.open=()=>null;change(document.querySelector('[aria-label="Phenotyping method"]'),'flow');
 await until(()=>button('Dock below',document));button('Dock below',document).click();
 await until(()=>app()?.querySelector('.flow-plot-foot')?.textContent.includes('Exact counts'));
 const card=app().querySelector('.flow-plot-card'),chart=card.querySelector('.flow-chart');
 change(card.querySelector('[aria-label="X measurement"]'),'M0');change(card.querySelector('[aria-label="Y measurement"]'),'M1');
 change(card.querySelector('[aria-label="Plot type"]'),'scatter');
 await until(()=>chart.data?.[0]?.type==='scattergl'&&chart.querySelector('[aria-label="Change X axis"]'));
 button('Expand',card).click();await delay(150);
 for(const axis of ['X','Y']){const title=chart.querySelector('[aria-label="Change '+axis+' axis"]'),r=title.getBoundingClientRect(),hit=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2);
   check(hit===title||title.contains(hit),axis+' axis title receives clicks at its visible position');}
 const axisChoice=async(axis,value)=>{const title=chart.querySelector('[aria-label="Change '+axis+' axis"]'),r=title.getBoundingClientRect();document.elementFromPoint(r.left+r.width/2,r.top+r.height/2).dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:r.left+r.width/2,clientY:r.top+r.height/2}));
   await until(()=>app().querySelector('.flow-axis-menu'));button(value,app().querySelector('.flow-axis-menu')).click();};
 await axisChoice('X','M1');await until(()=>chart.layout.xaxis.title.text.includes('M1'));
 check(card.querySelector('[aria-label="X measurement"]').value==='M1','Clicking the X-axis title changes the measurement and toolbar together');
 await axisChoice('X','M0');await until(()=>chart.layout.xaxis.title.text.includes('M0'));
 await axisChoice('Y','Count');await until(()=>chart.data?.[0]?.type==='scatter'&&chart.layout.yaxis.title.text==='Cells / bin');
 check(card.querySelector('[aria-label="Plot type"]').value==='histogram'&&chart.data[0].y.reduce((a,b)=>a+b,0)===40,'Y-axis Count switches to a histogram with exact cell counts');
 await axisChoice('Y','Proportion');await until(()=>chart.layout.yaxis.title.text==='Proportion / bin');
 check(Math.abs(chart.data[0].y.reduce((a,b)=>a+b,0)-1)<1e-12,'Y-axis Proportion displays fractions, not percentages');
 await axisChoice('Y','Percent');await until(()=>chart.layout.yaxis.title.text==='Cells / bin (%)');
 check(Math.abs(chart.data[0].y.reduce((a,b)=>a+b,0)-100)<1e-10,'Y-axis Percent displays percent per bin');
 await axisChoice('Y','M1');await until(()=>chart.data?.[0]?.type==='scattergl'&&chart.layout.yaxis.title.text.includes('M1'));
 check(card.querySelector('[aria-label="Plot type"]').value==='scatter','Choosing a Y measurement returns to a two-marker plot');
 check(chart.layout.xaxis.showline&&chart.layout.xaxis.mirror&&chart.layout.yaxis.showline&&chart.layout.yaxis.mirror,'Plots have a complete subtle frame on all four sides');
 button('Expand',card).click();
 let state=await info();window.__state.clusterReview={table:state.workspace.table,column:'phenotype_ml_consensus',colours:{A:'#ff8800',B:'#4433aa'}};
 await delay(1000);
 const beforeSplit=(await info()).workspace.revision;
 await splitBy('root','phenotype_ml_consensus',false);
 let categoryDialog=app().querySelector('dialog[open]');
 check((await info()).workspace.revision===beforeSplit&&(await info()).workspace.gates.length===0,'Categorical split opens a chooser without creating populations');
 check(categoryDialog.querySelectorAll('.flow-split-categories input:checked').length===2&&[...categoryDialog.querySelectorAll('.flow-split-categories .flow-number')].every(e=>e.textContent==='20'),'Category chooser starts with all values selected and exact parent counts');
 button('Cancel',categoryDialog).click();await until(()=>!app().querySelector('dialog[open]'));
 check((await info()).workspace.revision===beforeSplit,'Cancelling a categorical split leaves the hierarchy unchanged');
 await splitBy('root','phenotype_ml_consensus',false);categoryDialog=app().querySelector('dialog[open]');
 categoryDialog.querySelector('[aria-label="Keep B"]').click();await confirmSplit();
 state=await info();
 check(state.workspace.gates.length===1&&state.workspace.gates[0].name==='phenotype_ml_consensus_A'&&state.stats.find(s=>s.id==='root').n===40,'Only checked categories become subgroups; all cells remain in the parent');
 await splitBy('root','phenotype_ml_consensus',false);categoryDialog=app().querySelector('dialog[open]');
 button('None',categoryDialog).click();
 check(categoryDialog.querySelectorAll('.flow-split-categories input:checked').length===0&&categoryDialog.textContent.includes('0 / 2 selected'),'None clears the category selection and updates its summary');
 const noSelectionSerial=splitSerial;button('Split',categoryDialog).click();await until(()=>categoryDialog.querySelector('.flow-error').textContent.includes('Choose at least one'));
 check(splitSerial===noSelectionSerial&&(await info()).workspace.gates.length===1,'An empty category selection keeps the dialog open without submitting');
 button('All',categoryDialog).click();
 check(categoryDialog.querySelectorAll('.flow-split-categories input:checked').length===2,'All restores every category');
 button('None',categoryDialog).click();const categorySearch=categoryDialog.querySelector('[aria-label="Find split category"]');categorySearch.value='b';categorySearch.dispatchEvent(new Event('input'));
 check(categoryDialog.querySelector('[aria-label="Keep A"]').parentElement.hidden&&!categoryDialog.querySelector('[aria-label="Keep B"]').parentElement.hidden,'Category search narrows the visible checklist');
 categoryDialog.querySelector('[aria-label="Keep B"]').click();categorySearch.value='';categorySearch.dispatchEvent(new Event('input'));
 check(!categoryDialog.querySelector('[aria-label="Keep A"]').checked&&categoryDialog.querySelector('[aria-label="Keep B"]').checked,'Category selections survive clearing the search');
 await confirmSplit();
 await until(async()=>(await info()).workspace.gates.length===2);
 state=await info();const a=state.workspace.gates.find(g=>g.name==='phenotype_ml_consensus_A'),b=state.workspace.gates.find(g=>g.name==='phenotype_ml_consensus_B');
 check(!app().querySelector('dialog[open]')&&a&&b,'Confirming a different category adds its child without deleting earlier subgroups');
 check(a.colour==='#ff8800'&&b.colour==='#4433aa','Split colours match the Studio cluster-review palette');
 await until(()=>row(a.id)&&row(b.id));await splitBy(a.id,'tumor',false);
 check(app().querySelectorAll('dialog[open] .flow-split-categories input').length===2&&splitResponse.categories.map(c=>Number(c.value)).join(',')==='0,1','Low-cardinality numeric categories also open the chooser');
 await confirmSplit();
 await until(async()=>(await info()).workspace.gates.length===4);
 state=await info();const child=state.workspace.gates.find(g=>g.parent===a.id);
 await until(()=>row(child.id));
 check(state.stats.find(s=>s.id===child.id).n===10,'Low-cardinality numeric columns split into exact categorical subsets');
 check(parseInt(row(child.id).querySelector('.flow-tree-entry').style.paddingLeft)>parseInt(row(a.id).querySelector('.flow-tree-entry').style.paddingLeft),'Nested branches indent beneath their parent');
 await splitBy(a.id,'tumor');
 check(splitResponse.added===0&&(await info()).workspace.gates.length===4,'Repeating a split reuses existing children');
 row(a.id).querySelector('[type=checkbox]').click();
 row(b.id).querySelector('[type=checkbox]').click();
 await until(()=>chart.data?.length===2);
 row(child.id).querySelector('.flow-name').click();
 await until(()=>chart.data?.length===1&&window.__state.layers.some(l=>l.flowPopulation?.count===10));
 check(!app().querySelectorAll('.flow-tree input[type=checkbox]:checked').length,'Picking a branch clears the previous overlay comparison');
 showDataTab();await dataChoose('combined');await until(()=>dataView.queryReady&&dataView.result?.matched===10);
 const layer=window.__state.layers.find(l=>l.flowSelection);
 check(window.__state.layers.filter(l=>l.type==='cells').length===1&&layer.flowMode==='filter','Split selection reuses the segmentation layer and filters matching Data rows');
 const parent=card.querySelector('[aria-label="Plot population"]')||card.querySelector('.flow-plot-controls select');
 check(parent.selectedOptions[0].textContent==='phenotype_ml_consensus_A › '+child.name&&child.name.startsWith('tumor_'),'Plot population choices show prefixed names without repeating the column');
 // Both the plot and image consume the same saved swatch colour, even on cache hits.
 const swatch=row(child.id).querySelector('input[type=color]');check(!!swatch&&swatch.title.includes('Change colour'),'Each population has a labelled, clickable colour picker');
 change(swatch,'#12ab34');
 await until(()=>chart.data?.[0]?.marker?.color==='#12ab34'&&layer.flowPopulation?.colour==='#12ab34');
 check((await info()).workspace.gates.find(g=>g.id===child.id).colour==='#12ab34'&&dataView.result.matched===10,'Colour edits persist and sync plot/image without changing membership');
 row('root').querySelector('.flow-name').click();await until(()=>layer.flowPopulation?.name==='Analysis cells');
 change(row('root').querySelector('input[type=color]'),'#ab3412');
 await until(()=>chart.data?.[0]?.marker?.color==='#ab3412'&&layer.flowPopulation?.colour==='#ab3412');
 check((await info()).workspace.root_colour==='#ab3412','Analysis cells also has a saved colour shared by plot and viewer');
 button('Undo').click();await until(()=>row('root').querySelector('input[type=color]').value==='#a8b4c4');
 check(true,'Undo restores the previous root colour');
 // A second plot gets an independent population, leaving the first plot intact.
 button('New plot').click();await until(()=>app().querySelectorAll('.flow-plot-card').length===2);
 const second=app().querySelectorAll('.flow-plot-card')[1];
 await until(()=>second.querySelector('.flow-plot-foot').textContent.includes('Exact counts'));
 change(second.querySelector('.flow-plot-controls select'),b.id);
 await until(()=>layer.flowPopulation?.name==='phenotype_ml_consensus_B'&&dataView.result?.matched===20);
 check(card.querySelector('.flow-plot-controls select').value==='root','Changing one plot population leaves the other plot independent');
 await splitBy(b.id,'M0');await until(()=>app().querySelector('dialog[open]'));
 check((await info()).workspace.gates.length===4,'Continuous numeric columns ask for a condition before creating children');
 const dialog=app().querySelector('dialog');dialog.querySelector('[aria-label="Split cutoff"]').value=301;button('Split',dialog).click();
 await until(async()=>(await info()).workspace.gates.length===6);
 state=await info();const numeric=state.workspace.gates.filter(g=>g.parent===b.id);
 check(numeric.length===2&&numeric.every(g=>state.stats.find(s=>s.id===g.id).n===10),'Numeric threshold partitions the parent exactly at the cutoff');
 button('Undo').click();await until(async()=>(await info()).workspace.gates.length===4);
 check(true,'Undo removes a complete split in one step');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+app()?.querySelector('.flow-status')?.textContent));}})();
</script>'''

if __name__=='__main__':
    main(driver=DRIVER,studio=True,setup=setup,screenshot=False,timeout=180,browser_host='flow-gating.test')
